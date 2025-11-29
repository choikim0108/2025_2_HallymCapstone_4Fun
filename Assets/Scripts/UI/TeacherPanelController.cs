using UnityEngine;
using Photon.Pun;
using UnityEngine.UI;

public class TeacherPanelController : MonoBehaviourPunCallbacks
{
    public GameObject teacherPanel;
    public Button teacherPanelOpenButton;   // [변경] GameObject -> Button
    public Button teacherPanelCloseButton;  // [변경] GameObject -> Button

    void Start()
    {
        // 초기화
        OnCloseTeacherPanel();
        
        // 버튼 이벤트 안전하게 연결
        if (teacherPanelOpenButton != null)
        {
            teacherPanelOpenButton.onClick.RemoveAllListeners();
            teacherPanelOpenButton.onClick.AddListener(OnOpenTeacherPanel);
        }

        if (teacherPanelCloseButton != null)
        {
            teacherPanelCloseButton.onClick.RemoveAllListeners();
            teacherPanelCloseButton.onClick.AddListener(OnCloseTeacherPanel);
        }

        CheckRole();
    }

    void CheckRole()
    {
        object role;
        bool isTeacher = false;
        if (PhotonNetwork.LocalPlayer.CustomProperties.TryGetValue("Role", out role))
        {
            isTeacher = (role.ToString() == "Teacher");
        }

        if (!isTeacher)
        {
            if (teacherPanel != null) teacherPanel.SetActive(false);
            // Button 타입이므로 .gameObject를 통해 SetActive 사용
            if (teacherPanelOpenButton != null) teacherPanelOpenButton.gameObject.SetActive(false);
            if (teacherPanelCloseButton != null) teacherPanelCloseButton.gameObject.SetActive(false);
        }
    }

    public void OnOpenTeacherPanel()
    {
        if (teacherPanel != null) teacherPanel.SetActive(true);
        if (teacherPanelOpenButton != null) teacherPanelOpenButton.gameObject.SetActive(false);
        if (teacherPanelCloseButton != null) teacherPanelCloseButton.gameObject.SetActive(true);
    }

    public void OnCloseTeacherPanel()
    {
        if (teacherPanel != null) teacherPanel.SetActive(false);
        if (teacherPanelOpenButton != null) teacherPanelOpenButton.gameObject.SetActive(true);
        if (teacherPanelCloseButton != null) teacherPanelCloseButton.gameObject.SetActive(false);
    }

    // (녹음 및 NPC 셔플 기능은 그대로 유지)
    public void OnRecordStartButtonClicked()
    {
        var recorderObjs = Object.FindObjectsByType<VoiceRecorder>(FindObjectsSortMode.None);
        if (recorderObjs.Length > 0)
        {
            foreach (var recorder in recorderObjs)
            {
                var photonView = recorder.GetComponent<PhotonView>();
                if (photonView != null) photonView.RPC("RpcStartRecording", RpcTarget.All);
            }
        }
    }

    public void OnRecordStopButtonClicked()
    {
        var recorderObjs = Object.FindObjectsByType<VoiceRecorder>(FindObjectsSortMode.None);
        if (recorderObjs != null)
        {
            foreach (var recorder in recorderObjs)
            {
                var photonView = recorder.GetComponent<PhotonView>();
                if (photonView != null) photonView.RPC("RpcStopRecordingAndSave", RpcTarget.All);
            }
        }
    }

    public void OnNpcShuffleButtonClicked()
    {
        if (NpcManager.Instance != null) NpcManager.Instance.ShuffleNpcs();
    }
}