using Photon.Pun;
using Photon.Realtime;
using UnityEngine;
using UnityEngine.UI;
using TMPro; 
using System.Text; 

public class RoomInfoController : MonoBehaviourPunCallbacks
{
    [Header("TabKeyPanel UI")]
    public GameObject TabKeyPanelUI; // TAB 누르면 켤/끌 패널

    [Header("Sub Panels (For Reset)")] // 초기화를 위해 패널 연결 필요
    public GameObject TeacherPanel;
    public GameObject SetupPanel;

    [Header("Buttons on Tab Panel")]    // 토글할 버튼들
    public GameObject TeacherPanelOpenButton;
    public GameObject SetupPanelOpenButton;
    
    [Header("New Menu Buttons")] // 로비 이동 및 게임 종료 버튼
    public Button leaveButton;
    public Button quitButton;

    [Header("Camera Settings")] // 카메라 고정을 위한 변수
    public MonoBehaviour cameraMovementScript; 
    
    [Header("UI Texts")]
    public TextMeshProUGUI roomNameText; 
    public TextMeshProUGUI playerListText; 
    
    private bool isTabPanelActive = false; 

    void Start()
    {
        // 패널 초기화
        if (TabKeyPanelUI != null) TabKeyPanelUI.SetActive(false);
        // 시작 시 버튼은 안보이게 (TAB 눌러야 보임)
        if (TeacherPanelOpenButton != null) TeacherPanelOpenButton.SetActive(false);
        if (SetupPanelOpenButton != null) SetupPanelOpenButton.SetActive(false);

        // 버튼 기능 연결
        if (leaveButton != null) leaveButton.onClick.AddListener(OnLeaveRoomButtonClicked);
        if (quitButton != null) quitButton.onClick.AddListener(OnQuitGameButtonClicked);

        if (roomNameText != null && PhotonNetwork.CurrentRoom != null)
        {
            roomNameText.text = "방 이름: " + PhotonNetwork.CurrentRoom.Name;
        }

        if (cameraMovementScript != null) 
        {
            cameraMovementScript.enabled = true;
        }
    }

    void Update()
    {
        // CursorController에서 입력을 처리하므로 여기는 비워둡니다.
    }

    // CursorController가 호출할 공개 토글 함수
    public void ToggleTabPanel()
    {
        // 상태 반전
        isTabPanelActive = !isTabPanelActive;

        // 탭을 켜거나 끌 때, 하위 패널들을 강제로 '초기 상태(닫힘)'로 리셋합니다.
        // 이렇게 하면 버튼들은 다시 활성화되고 패널은 닫힙니다.
        if (SetupPanel != null)
        {
            var controller = SetupPanel.GetComponent<SetupPanelController>();
            if (controller != null) controller.ClosePanel();
        }

        if (TeacherPanel != null)
        {
            var controller = TeacherPanel.GetComponent<TeacherPanelController>();
            if (controller != null) controller.OnCloseTeacherPanel();
        }

        // 1. 메인 패널 토글
        if (TabKeyPanelUI != null)
        {
            TabKeyPanelUI.SetActive(isTabPanelActive);
        }

        // 2. Setup 버튼 토글 (위에서 ClosePanel()이 호출되어 버튼이 켜졌겠지만, 메인 패널 상태에 맞춥니다)
        if (SetupPanelOpenButton != null)
        {
            // 탭 패널이 켜질 때만 버튼을 보여줍니다. (닫히면 부모가 꺼지니 자동 숨김되지만 확실하게 처리)
            SetupPanelOpenButton.SetActive(isTabPanelActive);
        }

        // 3. Teacher 버튼 토글 (Teacher 역할 확인)
        if (TeacherPanelOpenButton != null)
        {
            object role;
            bool isTeacher = false;
            if (PhotonNetwork.LocalPlayer.CustomProperties.TryGetValue("Role", out role))
            {
                isTeacher = (role.ToString() == "Teacher");
            }

            // 탭 패널이 켜져있고, 선생님일 때만 버튼을 보여줍니다.
            TeacherPanelOpenButton.SetActive(isTabPanelActive && isTeacher);
        }

        // 4. 카메라 고정/해제
        if (cameraMovementScript != null)
        {
            cameraMovementScript.enabled = !isTabPanelActive;
        }

        // 5. 패널이 켜질 때만 목록 갱신
        if (isTabPanelActive)
        {
            UpdatePlayerList();
        }
    }

    public void OnLeaveRoomButtonClicked()
    {
        Debug.Log("방 떠나기 요청...");
        PhotonNetwork.LeaveRoom(); 
    }

    public void OnQuitGameButtonClicked()
    {
        Debug.Log("게임 종료...");
        Application.Quit(); 
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false; 
#endif
    }

    public override void OnLeftRoom()
    {
        Debug.Log("방을 나갔습니다. 로비 씬으로 이동합니다.");
        PhotonNetwork.LoadLevel("LobbyScene"); 
    }
    
    public bool IsPanelActive()
    {
        return isTabPanelActive; 
    }

    void UpdatePlayerList()
    {
        if (playerListText == null) return;

        StringBuilder sb = new StringBuilder();
        sb.AppendLine("참가 플레이어:");

        foreach (Player player in PhotonNetwork.PlayerList)
        {
            sb.AppendLine("- " + player.NickName);
        }

        playerListText.text = sb.ToString();
    }

    public override void OnEnable()
    {
        base.OnEnable(); 
        PhotonNetwork.AddCallbackTarget(this); 
    }

    public override void OnDisable()
    {
        base.OnDisable(); 
        PhotonNetwork.RemoveCallbackTarget(this); 
    }

    public override void OnPlayerEnteredRoom(Player newPlayer)
    {
        if (TabKeyPanelUI != null && TabKeyPanelUI.activeSelf) UpdatePlayerList();
    }

    public override void OnPlayerLeftRoom(Player otherPlayer)
    {
        if (TabKeyPanelUI != null && TabKeyPanelUI.activeSelf) UpdatePlayerList();
    }
}