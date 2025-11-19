using UnityEngine;
using Photon.Pun;
using UnityEngine.UI;

public class TeacherPanelController : MonoBehaviourPunCallbacks
{
    public GameObject teacherPanel;
    public GameObject teacherPanelOpenButton;
    public GameObject teacherPanelCloseButton;

    void Start()
    {
        // 내 플레이어의 CustomProperties에서 Role 확인
        object role;
        if (PhotonNetwork.LocalPlayer.CustomProperties.TryGetValue("Role", out role))
        {
            if (role.ToString() == "Teacher")
            {
                teacherPanel.SetActive(false);
                teacherPanelOpenButton.SetActive(false);
                teacherPanelCloseButton.SetActive(false);
            }
            else
            {
                teacherPanel.SetActive(false);
                teacherPanelOpenButton.SetActive(false);
                teacherPanelCloseButton.SetActive(false);
            }
        }
        else
        {
            teacherPanel.SetActive(false);
            teacherPanelOpenButton.SetActive(false);
            teacherPanelCloseButton.SetActive(false);
        }
    }

    // Open 버튼 클릭 시 호출
    public void OnOpenTeacherPanel()
    {
        teacherPanel.SetActive(true);
        teacherPanelOpenButton.SetActive(false);
        teacherPanelCloseButton.SetActive(true);
    }

    // Close 버튼 클릭 시 호출
    public void OnCloseTeacherPanel()
    {
        teacherPanel.SetActive(false);
        teacherPanelOpenButton.SetActive(true);
        teacherPanelCloseButton.SetActive(false);
    }

    public void OnRecordStartButtonClicked()
    {
        var recorderObjs = Object.FindObjectsByType<VoiceRecorder>(FindObjectsSortMode.None);
        if (recorderObjs.Length > 0)
        {
            foreach (var recorder in recorderObjs)
            {
                var photonView = recorder.GetComponent<PhotonView>();
                if (photonView != null)
                {
                    photonView.RPC("RpcStartRecording", RpcTarget.All);
                }
            }
            Debug.Log("RecordStartButtonClicked: VoiceRecorder 오브젝트를 찾았습니다.");
        }
        else
        {
            Debug.LogError("VoiceRecorder 오브젝트를 찾을 수 없습니다.");
        }
    }

    public void OnRecordStopButtonClicked()
    {
        // 모든 플레이어에게 녹음 중지 및 저장 명령 (Teacher 제외)
        var recorderObjs = Object.FindObjectsByType<VoiceRecorder>(FindObjectsSortMode.None);
        if (recorderObjs != null)
        {
            foreach (var recorder in recorderObjs)
            {
                var photonView = recorder.GetComponent<PhotonView>();
                if (photonView != null)
                {
                    photonView.RPC("RpcStopRecordingAndSave", RpcTarget.All);
                }
            }
        }
        else
        {
            Debug.LogError("VoiceRecorder 오브젝트를 찾을 수 없습니다.");
        }
        Debug.Log("RecordStopButtonClicked: VoiceRecorder 오브젝트를 찾았습니다.");
    }
}
