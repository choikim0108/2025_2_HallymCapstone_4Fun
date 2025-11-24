using Photon.Pun;
using Photon.Realtime;
using UnityEngine;
using UnityEngine.UI;
using TMPro; // TextMeshPro 사용
using System.Text; // StringBuilder 사용

public class RoomInfoController : MonoBehaviourPunCallbacks
{
    [Header("TabKeyPanel UI")]
    public GameObject TabKeyPanelUI; // TAB 누르면 켤/끌 패널
    [Header("Buttons on Tab Panel")]    // 토글할 버튼들
    public GameObject TeacherPanelOpenButton;
    public GameObject SetupPanelOpenButton;
    [Header("New Menu Buttons")] // 로비 이동 및 게임 종료 버튼
    public Button leaveButton;
    public Button quitButton;

    [Header("Camera Settings")] // 카메라 고정을 위한 변수
    public MonoBehaviour cameraMovementScript; // 카메라를 움직이는 스크립트 (예: MouseLook)    
    [Header("UI Texts")]
    public TextMeshProUGUI roomNameText; // 방 이름을 표시할 텍스트
    public TextMeshProUGUI playerListText; // 플레이어 목록을 표시할 텍스트
    private bool isTabPanelActive = false; // 탭 패널의 현재 활성화 상태

    void Start()
    {
        // 패널 초기화
        if (TabKeyPanelUI != null) TabKeyPanelUI.SetActive(false);
        if (TeacherPanelOpenButton != null) TeacherPanelOpenButton.SetActive(false);
        if (SetupPanelOpenButton != null) SetupPanelOpenButton.SetActive(false);

        // [추가] 버튼 기능 연결
        if (leaveButton != null)
            leaveButton.onClick.AddListener(OnLeaveRoomButtonClicked);
        
        if (quitButton != null)
            quitButton.onClick.AddListener(OnQuitGameButtonClicked);

        // 방 이름 설정 (한 번만 하면 됨)
        if (roomNameText != null && PhotonNetwork.CurrentRoom != null)
        {
            roomNameText.text = "방 이름: " + PhotonNetwork.CurrentRoom.Name;
        }
    }

void Update()
    {
/*
        // TAB 키를 '누르는 순간' (GetKeyDown)
        if (Input.GetKeyDown(KeyCode.Tab))
        {
            if (TabKeyPanelUI != null)
            {
                TabKeyPanelUI.SetActive(true);
                UpdatePlayerList(); // 켤 때만 목록 갱신
            }
        }
        
        // TAB 키에서 '손을 떼는 순간' (GetKeyUp)
        if (Input.GetKeyUp(KeyCode.Tab))
        {
            if (TabKeyPanelUI != null)
            {
                TabKeyPanelUI.SetActive(false);
            }
        }
*/
    }
// CursorController가 호출할 공개 토글 함수
    public void ToggleTabPanel()
    {
        // 상태 반전
        isTabPanelActive = !isTabPanelActive;

        // 메인 패널 토글
        if (TabKeyPanelUI != null)
        {
            TabKeyPanelUI.SetActive(isTabPanelActive);
        }

        // Setup 버튼 토글
        if (SetupPanelOpenButton != null)
        {
            SetupPanelOpenButton.SetActive(isTabPanelActive);
        }

        // Teacher 버튼 토글 (역할 확인)
        if (TeacherPanelOpenButton != null)
        {
            object role;
            bool isTeacher = false;
            if (PhotonNetwork.LocalPlayer.CustomProperties.TryGetValue("Role", out role))
            {
                isTeacher = (role.ToString() == "Teacher");
            }

            // 패널이 활성화 상태이고, 플레이어가 Teacher일 때만 버튼을 켬
            TeacherPanelOpenButton.SetActive(isTabPanelActive && isTeacher);
        }

        // 카메라 고정/해제 로직
        // UI가 켜지면(True) -> 카메라 스크립트 끄기 (False)
        // UI가 꺼지면(False) -> 카메라 스크립트 켜기 (True)
        if (cameraMovementScript != null)
        {
            cameraMovementScript.enabled = !isTabPanelActive;
        }

        // 패널이 켜질 때만 목록 갱신
        if (isTabPanelActive)
        {
            UpdatePlayerList();
        }

        // 패널이 켜질 때만 목록 갱신
        if (isTabPanelActive)
        {
            UpdatePlayerList();
        }
    }

    // [추가] 로비로 돌아가기 버튼 클릭 시
    public void OnLeaveRoomButtonClicked()
    {
        Debug.Log("방 떠나기 요청...");
        PhotonNetwork.LeaveRoom(); // 방 나가기 -> OnLeftRoom 호출됨
    }

    // [추가] 게임 종료 버튼 클릭 시
    public void OnQuitGameButtonClicked()
    {
        Debug.Log("게임 종료...");
        Application.Quit(); // 빌드된 게임 종료
        
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false; // 에디터에서도 종료되도록 함
#endif
    }

    // [추가] 방을 완전히 나갔을 때 호출되는 콜백
    public override void OnLeftRoom()
    {
        Debug.Log("방을 나갔습니다. 로비 씬으로 이동합니다.");
        // "LobbyScene" 부분을 실제 로비 씬 이름으로 변경해주세요.
        PhotonNetwork.LoadLevel("LobbyScene"); 
    }
    
    // CursorController가 현재 패널 상태를 확인할 수 있도록 함
    public bool IsPanelActive()
    {
        return isTabPanelActive; // 또는 TabKeyPanelUI.activeSelf
    }
    // 플레이어 목록을 갱신하는 함수
    void UpdatePlayerList()
    {
        if (playerListText == null) return;

        // StringBuilder를 사용하면 문자열 합칠 때 더 효율적입니다.
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("참가 플레이어:");

        // PhotonNetwork.PlayerList에는 현재 방에 있는 모든 플레이어 정보가 들어있습니다.
        // 이 리스트는 Photon에 의해 자동으로 갱신됩니다.
        foreach (Player player in PhotonNetwork.PlayerList)
        {
            sb.AppendLine("- " + player.NickName);
            // 만약 마스터 클라이언트를 표시하고 싶다면:
            // if (player.IsMasterClient)
            // {
            //     sb.AppendLine($"- {player.NickName} (방장)");
            // }
            // else
            // {
            //     sb.AppendLine($"- {player.NickName}");
            // }
        }

        playerListText.text = sb.ToString();
    }

    // --- Photon Callback 메소드 ---
    // (중요) 이 스크립트가 콜백을 받으려면 OnEnable/OnDisable에서 등록/해제해야 합니다.

    public override void OnEnable()
    {
        base.OnEnable(); // MonoBehaviourPunCallbacks의 OnEnable 호출
        PhotonNetwork.AddCallbackTarget(this); // 콜백 타겟으로 이 스크립트 추가
    }

    public override void OnDisable()
    {
        base.OnDisable(); // MonoBehaviourPunCallbacks의 OnDisable 호출
        PhotonNetwork.RemoveCallbackTarget(this); // 콜백 타겟에서 이 스크립트 제거
    }

    // 다른 플레이어가 방에 들어왔을 때 호출
    public override void OnPlayerEnteredRoom(Player newPlayer)
    {
        // 스코어보드 패널이 활성화되어 있을 때만 목록 갱신
        if (TabKeyPanelUI != null && TabKeyPanelUI.activeSelf)
        {
            UpdatePlayerList();
        }
    }

    // 다른 플레이어가 방에서 나갔을 때 호출
    public override void OnPlayerLeftRoom(Player otherPlayer)
    {
        // 스코어보드 패널이 활성화되어 있을 때만 목록 갱신
        if (TabKeyPanelUI != null && TabKeyPanelUI.activeSelf)
        {
            UpdatePlayerList();
        }
    }
}