
#pragma once

#include "Source/DeploymentThreads/DeploymentThread.h"

class PluginDeploymentThread: public DeploymentThread {
public:
    // initialize your deployment thread here
    PluginDeploymentThread():DeploymentThread() {

    }

    // this method runs on a per-event basis.
    // the majority of the deployment will be done here!
    std::pair<bool, bool> deploy (
        std::optional<MidiFileEvent> & new_midi_event_dragdrop,
        std::optional<EventFromHost> & new_event_from_host,
        bool gui_params_changed_since_last_call,
        bool new_preset_loaded_since_last_call,
        bool new_midi_file_dropped_on_visualizers,
        bool new_audio_file_dropped_on_visualizers) override {


        // Try loading the model if it hasn't been loaded yet
        if (!isModelLoaded) {
            load("drumLoopVAE.pt");
        }

        // Check if voice map should be updated
        bool voiceMapChanged = false;
        if (gui_params_changed_since_last_call) {
            voiceMapChanged = updateVoiceMap();
        }

        // check if a new midi file has been dropped on the visualizers
        bool shouldEncodeGroove = updateInputFromMidiFile();

        // check if density has been updated
        if (gui_params.wasParamUpdated("Density")) {
            density = gui_params.getValueFor("Density");
            shouldEncodeGroove = true;
        }

        // encode the groove if necessary
        if (shouldEncodeGroove) {
            if (isModelLoaded) {
                encodeGroove();
                generatePattern();
            }
        }

        // if the voice map has changed, or a new pattern has been generated,
        // prepare the playback sequence
        if ((voiceMapChanged || shouldEncodeGroove) && isModelLoaded) {
            preparePlaybackSequence();
            preparePlaybackPolicy();
            return {true, true};
        }

        // your implementation goes here
        return {false, false};
    }

private:
    // density of tge pattern to be generated
    float density = 0.5f;

    // following will be used for the inference
    // contains the above information in a single tensor of shape (1, 32, 27)
    torch::Tensor groove_hvo = torch::zeros({1, 32, 27}, torch::kFloat32);

    // add any member variables or methods you need here
    torch::Tensor latent_vector = torch::zeros({1, 128}, torch::kFloat32);
    torch::Tensor voice_thresholds = torch::ones({ 9 }, torch::kFloat32) * 0.5f;
    torch::Tensor max_counts_allowed = torch::ones({ 9 }, torch::kFloat32) * 32;
    int sampling_mode = 0;
    float temperature = 1.0f;
    std::map<int, int> voiceMap;

    torch::Tensor hits;
    torch::Tensor velocities;
    torch::Tensor offsets;

    // checks if any of the voice map parameters have been updated and updates the voice map
    // returns true if the voice map has been updated
    bool updateVoiceMap() {
        bool voiceMapChanged = false;
        if (gui_params.wasParamUpdated("Kick")) {
            voiceMap[0] = int(gui_params.getValueFor("Kick"));
            voiceMapChanged = true;
        }
        if (gui_params.wasParamUpdated("Snare")) {
            voiceMap[1] = int(gui_params.getValueFor("Snare"));
            voiceMapChanged = true;
        }
        if (gui_params.wasParamUpdated("ClosedHat")) {
            voiceMap[2] = int(gui_params.getValueFor("ClosedHat"));
            voiceMapChanged = true;
        }
        if (gui_params.wasParamUpdated("OpenHat")) {
            voiceMap[3] = int(gui_params.getValueFor("OpenHat"));
            voiceMapChanged = true;
        }
        if (gui_params.wasParamUpdated("LowTom")) {
            voiceMap[4] = int(gui_params.getValueFor("LowTom"));
            voiceMapChanged = true;
        }
        if (gui_params.wasParamUpdated("MidTom")) {
            voiceMap[5] = int(gui_params.getValueFor("MidTom"));
            voiceMapChanged = true;
        }
        if (gui_params.wasParamUpdated("HighTom")) {
            voiceMap[6] = int(gui_params.getValueFor("HighTom"));
            voiceMapChanged = true;
        }
        if (gui_params.wasParamUpdated("Crash")) {
            voiceMap[7] = int(gui_params.getValueFor("Crash"));
            voiceMapChanged = true;
        }
        if (gui_params.wasParamUpdated("Ride")) {
            voiceMap[8] = int(gui_params.getValueFor("Ride"));
            voiceMapChanged = true;
        }
        return voiceMapChanged;
    }

    // checks if a new midi file has been dropped on the visualizers and
    // updates the input tensor accordingly
    bool updateInputFromMidiFile() {
        // check if a midi file has been dropped on the visualizers
        auto new_sequence = midiVisualizersData->get_visualizer_data("MidiDropWidget");

        // return false if no new sequence is available
        if (new_sequence == std::nullopt) {
            return false;
        }
        
        // clear out existing groove input
        auto groove_hits = torch::zeros({1, 32, 1}, torch::kFloat32);
        auto groove_velocities = torch::zeros({1, 32, 1}, torch::kFloat32);
        auto groove_offsets = torch::zeros({1, 32, 1}, torch::kFloat32);
        for (const auto& event : *new_sequence) {
            if (event.isNoteOnEvent()) {
                // PrintMessage(event.getDescription().str());
                auto ppq  = event.Time(); // time in ppq
                auto velocity = event.getVelocity(); // velocity
                auto div = round(ppq / .25f);
                auto offset = (ppq - (div * .25f)) / 0.125 * 0.5 ;
                auto grid_index = (long long) fmod(div, 32);
                
                // check if louder if overlapping
                if (groove_hits[0][grid_index][0].item<float>() > 0) {
                    if (groove_velocities[0][grid_index][0].item<float>() < velocity) {
                        groove_velocities[0][grid_index][0] = velocity;
                        groove_offsets[0][grid_index][0] = offset;
                    }
                } else {
                    groove_hits[0][grid_index][0] = 1;
                    groove_velocities[0][grid_index][0] = velocity;
                    groove_offsets[0][grid_index][0] = offset;
                }
            }

        }

        // stack up the groove information into a single tensor
        groove_hvo = torch::concat(
            {
                groove_hits,
                groove_velocities,
                groove_offsets
            }, 2);

        // ignore the operations (I'm just changing the formation of the tensor)
        groove_hvo = torch::zeros({1, 32, 27});
        groove_hvo.index_put_(
            {torch::indexing::Ellipsis, 2},
            groove_hits.index({torch::indexing::Ellipsis, 0}));
        groove_hvo.index_put_(
            {torch::indexing::Ellipsis, 11},
            groove_velocities.index({torch::indexing::Ellipsis, 0}));
        groove_hvo.index_put_(
            {torch::indexing::Ellipsis, 20},
            groove_offsets.index({torch::indexing::Ellipsis, 0}));
        
        return true;
    }

    // encodes the groove into a latent vector using the encoder
    void encodeGroove() {
        // preparing the input to encode() method
        std::vector<torch::jit::IValue> enc_inputs;
        enc_inputs.emplace_back(groove_hvo);
        enc_inputs.emplace_back(torch::tensor(
                                    density,
                                    torch::kFloat32).unsqueeze_(0));

        // get the encode method
        auto encode = model.get_method("encode");

        // encode the input
        auto encoder_output = encode(enc_inputs);

        // get latent vector from encoder output
        latent_vector = encoder_output.toTuple()->elements()[2].toTensor();
    }

    // decodes a random latent vector into a pattern
    void generatePattern() {
        PrintMessage("Generating new sequence...");

        // Generate a random latent vector
        latent_vector = torch::randn({ 1, 128});

        // Prepare above for inference
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(latent_vector);
        inputs.emplace_back(voice_thresholds);
        inputs.emplace_back(max_counts_allowed);
        inputs.emplace_back(sampling_mode);
        inputs.emplace_back(temperature);

        // Get the scripted method
        auto sample_method = model.get_method("sample");

        // Run inference
        auto output = sample_method(inputs);

        // Extract the generated tensors from the output
        hits = output.toTuple()->elements()[0].toTensor();
        velocities = output.toTuple()->elements()[1].toTensor();
        offsets = output.toTuple()->elements()[2].toTensor();
    }

    // extracts the generated pattern into a PlaybackSequence
    void preparePlaybackSequence() {
        if (!hits.sizes().empty()) // check if any hits are available
        {
            // clear playback sequence
            playbackSequence.clear();

            // iterate through all voices, and time steps
            int batch_ix = 0;
            for (int step_ix = 0; step_ix < 32; step_ix++)
            {
                for (int voice_ix = 0; voice_ix < 9; voice_ix++)
                {

                    // check if the voice is active at this time step
                    if (hits[batch_ix][step_ix][voice_ix].item<float>() > 0.5)
                    {
                        auto midi_num = voiceMap[voice_ix];
                        auto velocity = velocities[batch_ix][step_ix][voice_ix].item<float>();
                        auto offset = offsets[batch_ix][step_ix][voice_ix].item<float>();
                        // we are going to convert the onset time to a ratio of quarter notes
                        auto time = (step_ix + offset) * 0.25f;

                        playbackSequence.addNoteWithDuration(
                            0, midi_num, velocity, time, 0.1f);

                    }
                }
            }
        }
    }

    // prepares the playback policy
    void preparePlaybackPolicy() {
        // Specify the playback policy
        playbackPolicy.SetPlaybackPolicy_RelativeToAbsoluteZero();
        playbackPolicy.SetTimeUnitIsPPQ();
        playbackPolicy.SetOverwritePolicy_DeleteAllEventsInPreviousStreamAndUseNewStream(true);
        playbackPolicy.ActivateLooping(8);
    }

};