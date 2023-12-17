#include "Source/DeploymentThreads/DeploymentThread.h"

// ===================================================================================
// ===         Refer to:
// https://neuralmidifx.github.io/DeploymentStages/????
// ===================================================================================
std::pair<bool, bool>
    DeploymentThread::deploy(
        std::optional<MidiFileEvent> & new_midi_event_dragdrop,
        std::optional<EventFromHost> & new_event_from_host,
        bool gui_params_changed_since_last_call,
        bool new_preset_loaded_since_last_call,
        bool new_midi_file_dropped_on_visualizers,
        bool new_audio_file_dropped_on_visualizers) {

    // flags to keep track if any new data is generated and should
    // be sent to the main thread
    bool newPlaybackPolicyShouldBeSent{false};
    bool newPlaybackSequenceGeneratedAndShouldBeSent{false};

    // =================================================================================
    // ===         LOADING THE MODEL IF IT HASN'T BEEN LOADED YET
    // =================================================================================
    // Try loading the model if it hasn't been loaded yet
    if (!isModelLoaded) {
        load("drumLoopVAE.pt");
    }

    if (new_midi_file_dropped_on_visualizers || gui_params_changed_since_last_call) {

        // =================================================================================
        // ===         1. Check (and Process)
        //              if there is a new midi file dropped on the visualizers
        // = Refer to:
        // https://neuralmidifx.github.io/docs/v2_0_0/datatypes/MidiVisualizersData
        // =================================================================================

        auto new_sequence = midiVisualizersData->get_visualizer_data("MidiDropWidget");
        if (new_sequence != std::nullopt) {

            // clear out existing groove input
            DPLdata.groove_hits = torch::zeros({1, 32, 1}, torch::kFloat32);
            DPLdata.groove_velocities = torch::zeros({1, 32, 1}, torch::kFloat32);
            DPLdata.groove_offsets = torch::zeros({1, 32, 1}, torch::kFloat32);
            for (const auto& event : *new_sequence) {
                if (event.isNoteOnEvent()) {
                    // PrintMessage(event.getDescription().str());
                    auto ppq  = new_midi_event_dragdrop->Time(); // time in ppq
                    auto velocity = new_midi_event_dragdrop->getVelocity(); // velocity
                    auto div = round(ppq / .25f);
                    auto offset = (ppq - (div * .25f)) / 0.125 * 0.5 ;
                    auto grid_index = (long long) fmod(div, 32);

                    // check if louder if overlapping
                    if (DPLdata.groove_hits[0][grid_index][0].item<float>() > 0) {
                        if (DPLdata.groove_velocities[0][grid_index][0].item<float>() < velocity) {
                            DPLdata.groove_velocities[0][grid_index][0] = velocity;
                            DPLdata.groove_offsets[0][grid_index][0] = offset;
                        }
                    } else {
                        DPLdata.groove_hits[0][grid_index][0] = 1;
                        DPLdata.groove_velocities[0][grid_index][0] = velocity;
                        DPLdata.groove_offsets[0][grid_index][0] = offset;
                    }
                }

            }

            // stack up the groove information into a single tensor
            DPLdata.groove_hvo = torch::concat(
                {
                    DPLdata.groove_hits,
                    DPLdata.groove_velocities,
                    DPLdata.groove_offsets
                }, 2);

            // ignore the operations (I'm just changing the formation of the tensor)
            DPLdata.groove_hvo = torch::zeros({1, 32, 27});
            DPLdata.groove_hvo.index_put_(
                {torch::indexing::Ellipsis, 2},
                DPLdata.groove_hits.index({torch::indexing::Ellipsis, 0}));
            DPLdata.groove_hvo.index_put_(
                {torch::indexing::Ellipsis, 11},
                DPLdata.groove_velocities.index({torch::indexing::Ellipsis, 0}));
            DPLdata.groove_hvo.index_put_(
                {torch::indexing::Ellipsis, 20},
                DPLdata.groove_offsets.index({torch::indexing::Ellipsis, 0}));

        }

        // create empty playback sequence if groove is empty (to avoid unnecessary inference)
        // don't run inference if groove is empty
        if (DPLdata.groove_hits.sum().item<float>() < 1.0f) {
            cout << "Groove is empty, skipping inference" << endl;
            playbackSequence.clear();
            newPlaybackSequenceGeneratedAndShouldBeSent = true;
            return {
                newPlaybackPolicyShouldBeSent,
                newPlaybackSequenceGeneratedAndShouldBeSent};
        }


        // ==============================================================================
        // ===              ACCESSING GUI PARAMETERS
        // Refer to:
        // https://neuralmidifx.github.io/docs/v2_0_0/datatypes/GuiParams
        // ==============================================================================
        auto density = gui_params.getValueFor("Density");

        // ==============================================================================
        // ===              Inference
        // ==============================================================================
        // WARNING always check if the model is loaded before running inference!!!
        if (isModelLoaded) {
            // ==========================================================================
            // ===              A. Prepare Inputs to Encoder to get Latent Vector
            // ==========================================================================
            // preparing the input to encode() method
            std::vector<torch::jit::IValue> enc_inputs;
            enc_inputs.emplace_back(DPLdata.groove_hvo);
            enc_inputs.emplace_back(torch::tensor(
                                        density,
                                        torch::kFloat32).unsqueeze_(0));

            // get the encode method
            auto encode = model.get_method("encode");

            // encode the input
            auto encoder_output = encode(enc_inputs);

            // get latent vector from encoder output
            DPLdata.latent_vector = encoder_output.toTuple()->elements()[2].toTensor();

            // ==========================================================================
            // ===              B. Decode/Sample using the Latent Vector to get the output
            // ==========================================================================

            // Prepare the inputs to the sample() method
            auto voice_thresholds = torch::ones({9 }, torch::kFloat32) * 0.5f;
            auto max_counts_allowed = torch::ones({9 }, torch::kFloat32) * 32;
            int sampling_mode = 0;
            float temperature = 1.0f;

            // Prepare above for inference
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(DPLdata.latent_vector);
            inputs.emplace_back(voice_thresholds);
            inputs.emplace_back(max_counts_allowed);
            inputs.emplace_back(sampling_mode);
            inputs.emplace_back(temperature);

            // Get the scripted method
            auto sample_method = model.get_method("sample");

            // Run inference
            auto output = sample_method(inputs);

            // Extract the generated tensors from the output
            auto hits = output.toTuple()->elements()[0].toTensor();
            auto velocities = output.toTuple()->elements()[1].toTensor();
            auto offsets = output.toTuple()->elements()[2].toTensor();

            // =========================================================================
            // ===              C. Extract Generations into a PlaybackPolicy and
            //                          PlaybackSequence
            // Refer to:
            // https://neuralmidifx.github.io/docs/v2_0_0/datatypes/PlaybackPolicy
            // https://neuralmidifx.github.io/docs/v2_0_0/datatypes/PlaybackSequence
            // =========================================================================

            // get the voice map
            std::map<int, int> voiceMap;
            voiceMap[0] = int(gui_params.getValueFor("Kick"));
            voiceMap[1] = int(gui_params.getValueFor("Snare"));
            voiceMap[2] = int(gui_params.getValueFor("ClosedHat"));
            voiceMap[3] = int(gui_params.getValueFor("OpenHat"));
            voiceMap[4] = int(gui_params.getValueFor("LowTom"));
            voiceMap[5] = int(gui_params.getValueFor("MidTom"));
            voiceMap[6] = int(gui_params.getValueFor("HighTom"));
            voiceMap[7] = int(gui_params.getValueFor("Crash"));
            voiceMap[8] = int(gui_params.getValueFor("Ride"));

            // clear playback sequence
            playbackSequence.clear();

            // set the flag to notify new playback sequence is generated
            newPlaybackSequenceGeneratedAndShouldBeSent = true;

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

            // Specify the playback policy
            playbackPolicy.SetPlaybackPolicy_RelativeToAbsoluteZero();
            playbackPolicy.SetTimeUnitIsPPQ();
            playbackPolicy.SetOverwritePolicy_DeleteAllEventsInPreviousStreamAndUseNewStream(true);
            playbackPolicy.ActivateLooping(8);
            newPlaybackPolicyShouldBeSent = true;

        }

    }

    /*// =================================================================================
    // ===         2. initialize latent vectors on the first call
    // =================================================================================
    if (DPLdata.latent_A.size(0) == 0) {
        DPLdata.latent_A = torch::randn({ 1, 128 });
    }
    if (DPLdata.latent_B.size(0) == 0) {
        DPLdata.latent_B = torch::randn({ 1, 128 });
    }

    // =================================================================================
    // ===         0. LOADING THE MODEL
    // =================================================================================
    // Try loading the model if it hasn't been loaded yet
    if (!isModelLoaded) {
        load("drumLoopVAE.pt");
    }

    // =================================================================================
    // ===         1. ACCESSING GUI PARAMETERS
    // Refer to:
    // https://neuralmidifx.github.io/docs/v1_0_0/datatypes/GuiParams#accessing-the-ui-parameters
    // =================================================================================

    bool should_interpolate = false;   // flag to check if we should interpolate

    // check if the buttons have been clicked, if so, update the DPLdata
    auto ButtonATriggered = gui_params.wasButtonClicked("Random A");
    if (ButtonATriggered) {
        should_interpolate = true;
        DPLdata.latent_A = torch::randn({ 1, 128 });
    }
    auto ButtonBTriggered = gui_params.wasButtonClicked("Random B");
    if (ButtonBTriggered) {
        should_interpolate = true;
        DPLdata.latent_B = torch::randn({ 1, 128 });
    }

    // check if the interpolate slider has changed, if so, update the DPLdata
    auto sliderValue = gui_params.getValueFor("Interpolate");
    bool sliderChanged = (sliderValue != DPLdata.interpolate_slider_value);
    if (sliderChanged) {
        should_interpolate = true;
        DPLdata.interpolate_slider_value = sliderValue;
    }

    // check if the preset has changed, if so, update the DPLdata
    if (new_preset_loaded_since_last_call) {
        should_interpolate = true;
        auto l_a = CustomPresetData->tensor("latent_A");
        auto l_b = CustomPresetData->tensor("latent_B");
        if (l_a != std::nullopt) {
            DPLdata.latent_A = *l_a;
        }
        if (l_b != std::nullopt) {
            DPLdata.latent_B = *l_b;
        }
    }

    if (should_interpolate) {

        if (isModelLoaded)
        {

            // calculate interpolated latent vector
            auto slider_value = DPLdata.interpolate_slider_value;
            auto latent_A = DPLdata.latent_A;
            auto latent_B = DPLdata.latent_B;
            auto latentVector = (1 - slider_value) * latent_A + slider_value * latent_B;

            // Backup the data for preset saving
            CustomPresetData->tensor("latent_A", latent_A);
            CustomPresetData->tensor("latent_B", latent_B);

            // Prepare other inputs
            auto voice_thresholds = torch::ones({9 }, torch::kFloat32) * 0.5f;
            auto max_counts_allowed = torch::ones({9 }, torch::kFloat32) * 32;
            int sampling_mode = 0;
            float temperature = 1.0f;

            // Prepare above for inference
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(latentVector);
            inputs.emplace_back(voice_thresholds);
            inputs.emplace_back(max_counts_allowed);
            inputs.emplace_back(sampling_mode);
            inputs.emplace_back(temperature);

            // Get the scripted method
            auto sample_method = model.get_method("sample");

            // Run inference
            auto output = sample_method(inputs);

            // Extract the generated tensors from the output
            auto hits = output.toTuple()->elements()[0].toTensor();
            auto velocities = output.toTuple()->elements()[1].toTensor();
            auto offsets = output.toTuple()->elements()[2].toTensor();

            // =================================================================================
            // ===         2. ACCESSING GUI PARAMETERS
            // Refer to:
            // https://neuralmidifx.github.io/docs/v2_0_0/datatypes/GuiParams
            // =================================================================================
            std::map<int, int> voiceMap;
            voiceMap[0] = int(gui_params.getValueFor("Kick"));
            voiceMap[1] = int(gui_params.getValueFor("Snare"));
            voiceMap[2] = int(gui_params.getValueFor("ClosedHat"));
            voiceMap[3] = int(gui_params.getValueFor("OpenHat"));
            voiceMap[4] = int(gui_params.getValueFor("LowTom"));
            voiceMap[5] = int(gui_params.getValueFor("MidTom"));
            voiceMap[6] = int(gui_params.getValueFor("HighTom"));
            voiceMap[7] = int(gui_params.getValueFor("Crash"));
            voiceMap[8] = int(gui_params.getValueFor("Ride"));


            // =================================================================================
            // ===         3. Extract Generations into a PlaybackPolicy and PlaybackSequence
            // Refer to:
            // https://neuralmidifx.github.io/docs/v2_0_0/datatypes/PlaybackPolicy
            // https://neuralmidifx.github.io/docs/v2_0_0/datatypes/PlaybackSequence
            // =================================================================================
            if (!hits.sizes().empty()) // check if any hits are available
            {
                // clear playback sequence
                playbackSequence.clear();

                // set the flag to notify new playback sequence is generated
                newPlaybackSequenceGeneratedAndShouldBeSent = true;

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

            // Specify the playback policy
            playbackPolicy.SetPlaybackPolicy_RelativeToAbsoluteZero();
            playbackPolicy.SetTimeUnitIsPPQ();
            playbackPolicy.SetOverwritePolicy_DeleteAllEventsInPreviousStreamAndUseNewStream(true);
            playbackPolicy.ActivateLooping(8);
            newPlaybackPolicyShouldBeSent = true;
        }
    }*/

    // your implementation goes here
    return {newPlaybackPolicyShouldBeSent, newPlaybackSequenceGeneratedAndShouldBeSent};
}
