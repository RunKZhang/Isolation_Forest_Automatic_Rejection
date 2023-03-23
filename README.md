# Isolation_Forest_Automatic_Rejection
A preprocessing tool for the automatic rejection of Long-Term EEG

## Install
This is a preprocessing tool developed for the automatic rejecton of Long-Term EEG, and those who are interested in the automatic preprocessing of Long-Term EEG can use it under MIT license.

To use the code, you should firstly:`git clone https://github.com/RunKZhang/Isolation_Forest_Automatic_Rejection.git`.

Then type the command `pip install -e .` in the root directory by installing it in editable mode.

The implementation of the code uses below packages:  **mne**,**mne_features**, **sklearn**, **numpy**.

## Usage
After installation, the package can be imported with:
`import IFReject`

Then the package can be used by using the follow codes in your scripts:

`if_class = IF_Reject.if_reject(epochs,['ptp_amp'])`

In the below code, *epochs* represents the epochs class of **mne** and *ptp_amp* represents peak-to-peak amplitude and it can be replaced by other features of EEG as described in **mne_features**.

Then the clean epochs should be obtained by using the following code:

`epochs_new = if_class.run()`
