# Description
The following repository contains a PyTorch-based recurrent neural network trained on the TIMIT dataset. It gives frame-level phones as output and is trained using CTC loss. The following architectures are implemented: LSTM, GRU, TCNN (and their bidirectional versions). A custom implementation of the recurrent neural networks is also provided which makes it easy to modify the core equations.

# Usage

For training:
1. Place the parent directory of TIMIT dataset in the _config.yaml_ file (config['dir']['dataset']).
2. Customise the hyperparameters in _config.yaml_.
2. Run the function _train_ in dl_model.py.

For inference:
1. Ensure that the model_configuration folder containt the pre-trained model where configuration is the following string:
<name_of_model>\_<number_of_layers>\_<number_of_hidden_units>\_<number_of_audio_features>
2. Run the function _infer_ in _dl_model.py_ with argument as a list of file paths of .wav files which are to be passed through the model.

Check the commented code at the very end in _dl_model.py_ for an illustration of how the results are generated.