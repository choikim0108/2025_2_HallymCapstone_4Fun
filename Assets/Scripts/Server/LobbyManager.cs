using Photon.Pun;
using Photon.Realtime;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

// MonoBehaviourPunCallbacks를 상속받아 포톤 관련 이벤트 콜백을 쉽게 사용
public class LobbyManager : MonoBehaviourPunCallbacks
{
    // 1. 싱글톤 인스턴스 추가: 다른 스크립트에서 쉽게 접근하기 위함
    public static LobbyManager Instance;

    private string gameVersion = "1"; // 게임 버전

    [Header("UI Panels")]
    public GameObject connectPanel; // 이제 이 패널은 거의 사용되지 않습니다.
    public GameObject lobbyPanel;
    public GameObject roomPanel;

    [Header("Lobby Panel")]
    public TMP_InputField roomNameInput;   // ★ 수정: InputField → TMP_InputField
    public Button createRoomButton;
    public Button joinRoomButton;
    public Button customizeButton;
    public Text lobbyStatusText;

    [Header("Room Panel")]
    public TextMeshProUGUI roomNameText;   // ★ 수정: Text → TextMeshProUGUI
    public TextMeshProUGUI playerListText; // ★ 수정: Text → TextMeshProUGUI
    public Button startGameButton;

    void Awake()
    {
        // 싱글톤 인스턴스 설정
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
        }

        // 모든 클라이언트의 씬을 자동으로 동기화
        PhotonNetwork.AutomaticallySyncScene = true;
    }

    void Start()
    {
        // UI 버튼에 기능 연결
        createRoomButton.onClick.AddListener(CreateRoom);
        joinRoomButton.onClick.AddListener(JoinRoom);
        if (customizeButton != null)
        {
            customizeButton.onClick.AddListener(OpenCustomizeScene);
        }

        // connectButton 관련 로직은 Firebase에서 바로 연결을 시도하므로 제거하거나 비활성화합니다.
        // connectButton.onClick.AddListener(ConnectToServer);

        // 시작 시 현재 Photon 연결 상태에 맞춰 UI 갱신
        InitializePanels();
    }
    
    // 닉네임을 매개변수로 받도록 수정
    public void ConnectToPhoton(string userId, string nickname)
    {
        if (PhotonNetwork.IsConnected)
        {
            Debug.Log("이미 서버에 연결되어 있습니다.");
            return;
        }

        Debug.Log($"Firebase ID({userId})로 포톤 서버에 접속을 시도합니다.");
        
        PhotonNetwork.AuthValues = new AuthenticationValues(userId);
        PhotonNetwork.NickName = nickname; // 전달받은 닉네임으로 Photon NickName 설정
        PhotonNetwork.GameVersion = gameVersion;
        PhotonNetwork.ConnectUsingSettings();
    }
    // 기존의 ConnectToServer 메서드는 이제 사용하지 않으므로 삭제하거나 주석 처리합니다.
    /*
    private void ConnectToServer()
    {
        // ... 기존 코드 ...
    }
    */
    
    // 방 생성
    private void CreateRoom()
    {
        if (string.IsNullOrEmpty(roomNameInput.text))
        {
            Debug.Log("방 이름을 입력하세요.");
            if (lobbyStatusText != null) lobbyStatusText.text = "방 이름을 입력하세요."; // [추가] 방 이름 미입력 시 UI 메시지 표시
            return;
        }

        if (lobbyStatusText != null) lobbyStatusText.text = "방 생성 중..."; // [추가] 요구조건 1: 방 생성 시도 시 UI 메시지 표시

        RoomOptions roomOptions = new RoomOptions();
        roomOptions.MaxPlayers = 4; //최대 참가인원 4명
        PhotonNetwork.CreateRoom(roomNameInput.text, roomOptions);
    }
    
    // 방 참여
    private void JoinRoom()
    {
        if (string.IsNullOrEmpty(roomNameInput.text))
        {
            Debug.Log("방 이름을 입력하세요.");
            if (lobbyStatusText != null) lobbyStatusText.text = "방 이름을 입력하세요."; // [추가] 방 이름 미입력 시 UI 메시지 표시
            return;
        }

        if (lobbyStatusText != null) lobbyStatusText.text = "방 참가 중..."; // [추가] 요구조건 1: 방 참가 시도 시 UI 메시지 표시

        PhotonNetwork.JoinRoom(roomNameInput.text);
    }   
    
    #region Photon Callback Methods
    
    // 마스터 서버에 접속했을 때 호출되는 콜백
    public override void OnConnectedToMaster()
    {
        Debug.Log("마스터 서버에 접속했습니다.");
        ShowLobbyPanel();
        UpdateCustomizeButtonState(true);
    CurrencyManager.TryPushCachedLoadoutToPhoton();
    }

    // 방 생성에 성공했을 때 호출되는 콜백
    public override void OnCreatedRoom()
    {
        Debug.Log("방 생성 성공!");
    }

    // [추가] 요구조건 2: 방 생성 실패 시 콜백
    public override void OnCreateRoomFailed(short returnCode, string message)
    {
        Debug.Log($"방 생성 실패: {message} (코드: {returnCode})");
        if (lobbyStatusText != null)
        {
            if (returnCode == 32766) // ErrorCode.GameIdAlreadyExists
            {
                lobbyStatusText.text = "이미 존재하는 방 이름입니다.";
            }
            else
            {
                lobbyStatusText.text = $"방 생성 실패: {message}";
            }
        }
    }

    // 방에 참여했을 때 호출되는 콜백
    public override void OnJoinedRoom()
    {
        Debug.Log("방에 참여했습니다.");
        ShowRoomPanel();

        // 마스터 클라이언트에게 teacher 역할 부여
        if (PhotonNetwork.IsMasterClient)
        {
            ExitGames.Client.Photon.Hashtable props = new ExitGames.Client.Photon.Hashtable();
            props["Role"] = "Teacher";
            PhotonNetwork.LocalPlayer.SetCustomProperties(props);
        }
        else
        {
            ExitGames.Client.Photon.Hashtable props = new ExitGames.Client.Photon.Hashtable();
            props["Role"] = "Student";
            PhotonNetwork.LocalPlayer.SetCustomProperties(props);
        }
    }

    // [추가] 요구조건 2: 방 참가 실패 시 콜백
    public override void OnJoinRoomFailed(short returnCode, string message)
    {
        Debug.Log($"방 참가 실패: {message} (코드: {returnCode})");
        if (lobbyStatusText != null)
        {
            if (returnCode == 32758) // ErrorCode.GameDoesNotExist
            {
                lobbyStatusText.text = "존재하지 않는 방입니다.";
            }
            else if (returnCode == 32765) // ErrorCode.GameFull
            {
                lobbyStatusText.text = "방이 가득 찼습니다.";
            }
            else
            {
                lobbyStatusText.text = $"참가 실패: {message}";
            }
        }
    }
    /*  10.01.12:43 146~151 line 수정
        // 다른 플레이어가 방에 참여했을 때 호출되는 콜백
        public override void OnPlayerEnteredRoom(Player newPlayer)
        {
            Debug.Log(newPlayer.NickName + "님이 참가했습니다.");
            UpdatePlayerList();
        }
    */
        // 다른 플레이어가 방에 참여했을 때 호출되는 콜백
        public override void OnPlayerEnteredRoom(Photon.Realtime.Player newPlayer) // <- Photon.Realtime 네임스페이스 명시
        {
            Debug.Log(newPlayer.NickName + "님이 참가했습니다.");
            UpdatePlayerList();
        }
    /*  10.01.12:43 161~165 line 수정
        // 플레이어가 방에서 나갔을 때 호출되는 콜백
        public override void OnPlayerLeftRoom(Player otherPlayer)
        {
            Debug.Log(otherPlayer.NickName + "님이 퇴장했습니다.");
            UpdatePlayerList();
        }
    */    
        // 플레이어가 방에서 나갔을 때 호출되는 콜백
        public override void OnPlayerLeftRoom(Photon.Realtime.Player otherPlayer) // <- Photon.Realtime 네임스페이스 명시
        {
            Debug.Log(otherPlayer.NickName + "님이 퇴장했습니다.");
            UpdatePlayerList();
        }
        
            public override void OnLeftRoom()
            {
                Debug.Log("방에서 나왔습니다.");
                ShowLobbyPanel();
            }
    #endregion

/*  10.01.12:43 179~188 line 수정
    private void UpdatePlayerList()
    {
        string playerListString = "플레이어 목록:\n";
        foreach (Player player in PhotonNetwork.PlayerList)
        {
            playerListString += player.NickName + "\n";
        }
        playerListText.text = playerListString;
    }
*/
    private void UpdatePlayerList()
    {
        string playerListString = "플레이어 목록:\n";
        // 이곳의 Player 타입을 명확히 지정해줍니다.
        foreach (Photon.Realtime.Player player in PhotonNetwork.PlayerList)
        {
            playerListString += player.NickName + "\n";
        }
        playerListText.text = playerListString;
    }

    public void OnStartGameButtonClicked()
    {
        if (PhotonNetwork.IsMasterClient)
        {
            PhotonNetwork.LoadLevel("MainScene");
        }
    }

    private void OpenCustomizeScene()
    {
        UnityEngine.SceneManagement.SceneManager.LoadScene("CustomizeScene");
    }

    private void InitializePanels()
    {
        if (PhotonNetwork.InRoom)
        {
            ShowRoomPanel();
            return;
        }

        if (PhotonNetwork.IsConnectedAndReady)
        {
            ShowLobbyPanel();
            UpdateCustomizeButtonState(true);
            CurrencyManager.TryPushCachedLoadoutToPhoton();
            return;
        }

        ShowConnectPanel();
        UpdateCustomizeButtonState(false);
    }

    private void ShowConnectPanel()
    {
        if (connectPanel != null) connectPanel.SetActive(true);
        if (lobbyPanel != null) lobbyPanel.SetActive(false);
        if (roomPanel != null) roomPanel.SetActive(false);
        UpdateCustomizeButtonState(false);
    }

    private void ShowLobbyPanel()
    {
        if (connectPanel != null) connectPanel.SetActive(false);
        if (lobbyPanel != null) lobbyPanel.SetActive(true);
        if (roomPanel != null) roomPanel.SetActive(false);

        if (lobbyStatusText != null) lobbyStatusText.text = ""; // [추가] 로비 패널로 돌아올 때 상태 메시지 초기화

        if (PhotonNetwork.IsConnectedAndReady && !PhotonNetwork.InLobby)
        {
            PhotonNetwork.JoinLobby();
        }

        UpdateCustomizeButtonState(true);
        CurrencyManager.TryPushCachedLoadoutToPhoton();
    }

    private void ShowRoomPanel()
    {
        if (connectPanel != null) connectPanel.SetActive(false);
        if (lobbyPanel != null) lobbyPanel.SetActive(false);
        if (roomPanel != null) roomPanel.SetActive(true);

        if (roomNameText != null && PhotonNetwork.CurrentRoom != null)
        {
            roomNameText.text = "방 이름: " + PhotonNetwork.CurrentRoom.Name;
        }

        if (startGameButton != null)
        {
            startGameButton.gameObject.SetActive(PhotonNetwork.IsMasterClient);
        }

        UpdatePlayerList();
    }

    private void UpdateCustomizeButtonState(bool canUse)
    {
        if (customizeButton != null)
        {
            customizeButton.interactable = canUse;
        }
    }
}