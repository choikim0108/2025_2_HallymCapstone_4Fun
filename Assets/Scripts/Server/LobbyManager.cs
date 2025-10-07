using Photon.Pun;
using Photon.Realtime;
using UnityEngine;
using UnityEngine.UI;

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
    public InputField roomNameInput;
    public Button createRoomButton;
    public Button joinRoomButton;

    [Header("Room Panel")]
    public Text roomNameText;
    public Text playerListText;
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

        // connectButton 관련 로직은 Firebase에서 바로 연결을 시도하므로 제거하거나 비활성화합니다.
        // connectButton.onClick.AddListener(ConnectToServer);

        // 시작 시 로비/룸 패널은 비활성화
        lobbyPanel.SetActive(false);
        roomPanel.SetActive(false);
        // connectPanel도 비활성화 처리. 연결 상태를 보여주는 텍스트 등으로 대체할 수 있습니다.
        connectPanel.SetActive(true);
    }
    
    // 2. Firebase ID로 포톤에 접속하는 public 메서드 (핵심!)
    public void ConnectToPhoton(string userId)
    {
        if (PhotonNetwork.IsConnected)
        {
            Debug.Log("이미 서버에 연결되어 있습니다.");
            return;
        }

        Debug.Log($"Firebase ID({userId})로 포톤 서버에 접속을 시도합니다.");
        
        // 가장 중요한 부분: Photon의 인증값으로 Firebase UserId를 사용
        PhotonNetwork.AuthValues = new AuthenticationValues(userId);
        PhotonNetwork.NickName = "Player_" + userId.Substring(0, 5); // 닉네임은 임시로 생성
        PhotonNetwork.GameVersion = gameVersion;
        PhotonNetwork.ConnectUsingSettings(); // 포톤 서버 접속 시작
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
            return;
        }
        RoomOptions roomOptions = new RoomOptions();
        roomOptions.MaxPlayers = 4; // 최대 4명까지
        PhotonNetwork.CreateRoom(roomNameInput.text, roomOptions);
    }
    
    // 방 참여
    private void JoinRoom()
    {
        if (string.IsNullOrEmpty(roomNameInput.text))
        {
            Debug.Log("방 이름을 입력하세요.");
            return;
        }
        PhotonNetwork.JoinRoom(roomNameInput.text);
    }
    
    #region Photon Callback Methods
    
    // 마스터 서버에 접속했을 때 호출되는 콜백
    public override void OnConnectedToMaster()
    {
        Debug.Log("마스터 서버에 접속했습니다.");
        connectPanel.SetActive(false);
        lobbyPanel.SetActive(true); // 접속 성공 시 로비 패널 활성화
    }

    // 방 생성에 성공했을 때 호출되는 콜백
    public override void OnCreatedRoom()
    {
        Debug.Log("방 생성 성공!");
    }

    // 방에 참여했을 때 호출되는 콜백
    public override void OnJoinedRoom()
    {
        Debug.Log("방에 참여했습니다.");
        lobbyPanel.SetActive(false);
        roomPanel.SetActive(true);
        roomNameText.text = "방 이름: " + PhotonNetwork.CurrentRoom.Name;
        UpdatePlayerList();
        
        startGameButton.gameObject.SetActive(PhotonNetwork.IsMasterClient);

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
}