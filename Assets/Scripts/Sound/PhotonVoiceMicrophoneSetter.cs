using UnityEngine;
using Photon.Voice.Unity;
using Photon.Voice;

public class PhotonVoiceMicrophoneSetter : MonoBehaviour
{
    public Recorder recorder;

    public void SetMicrophone(string micName)
    {
        if (!string.IsNullOrEmpty(micName) && recorder != null)
        {
            recorder.MicrophoneDevice = new DeviceInfo(micName);
            Debug.Log("[PhotonVoiceMicrophoneSetter] 마이크 설정됨: " + micName);
        }
    }
}