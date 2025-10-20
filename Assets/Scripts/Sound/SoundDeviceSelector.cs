using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class SoundDeviceSelector : MonoBehaviour
{
    public TMP_Dropdown microphoneDropdown;
    public Button applyButton;
    public TMP_Text selectedMicText;
    public TMP_Text currentMicText;
    public PhotonVoiceMicrophoneSetter microphoneSetter; // Inspector에서 연결

    public string SelectedMicrophone { get; private set; }

    void Start()
    {
        microphoneDropdown.ClearOptions();
        var micDevices = Microphone.devices;
        microphoneDropdown.AddOptions(new System.Collections.Generic.List<string>(micDevices));

        if (micDevices.Length > 0)
        {
            SelectedMicrophone = micDevices[0];
            selectedMicText.text = "선택된 마이크: " + SelectedMicrophone;
        }
        else
        {
            selectedMicText.text = "마이크 없음";
        }

        // 현재 설정된 마이크 표시
        string currentMic = PlayerPrefs.GetString("SelectedMicrophone", SelectedMicrophone);
        currentMicText.text = "현재 마이크: " + currentMic;

        applyButton.onClick.AddListener(OnApplyClicked);
        microphoneDropdown.onValueChanged.AddListener(OnMicDropdownChanged);
    }

    void OnMicDropdownChanged(int index)
    {
        SelectedMicrophone = Microphone.devices[index];
        selectedMicText.text = "선택된 마이크: " + SelectedMicrophone;

        // 마이크 변경을 실시간 반영
        if (microphoneSetter != null)
        {
            microphoneSetter.SetMicrophone(SelectedMicrophone);
            Debug.Log($"[SoundDeviceSelector] 마이크 변경됨: {SelectedMicrophone}");
        }
    }

    void OnApplyClicked()
    {
        PlayerPrefs.SetString("SelectedMicrophone", SelectedMicrophone);
        PlayerPrefs.Save();
        currentMicText.text = "현재 마이크: " + SelectedMicrophone;
    }
}
