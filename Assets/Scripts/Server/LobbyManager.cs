using Photon.Pun;
using Photon.Realtime;
using UnityEngine;
using UnityEngine.UI;

// MonoBehaviourPunCallbacks�� ��ӹ޾� ���� ���� �̺�Ʈ �ݹ��� ���� ���
public class LobbyManager : MonoBehaviourPunCallbacks
{
    // 1. �̱��� �ν��Ͻ� �߰�: �ٸ� ��ũ��Ʈ���� ���� �����ϱ� ����
    public static LobbyManager Instance;

    private string gameVersion = "1"; // ���� ����

    [Header("UI Panels")]
    public GameObject connectPanel; // ���� �� �г��� ���� ������ �ʽ��ϴ�.
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
        // �̱��� �ν��Ͻ� ����
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
        }

        // ��� Ŭ���̾�Ʈ�� ���� �ڵ����� ����ȭ
        PhotonNetwork.AutomaticallySyncScene = true;
    }

    void Start()
    {
        // UI ��ư�� ��� ����
        createRoomButton.onClick.AddListener(CreateRoom);
        joinRoomButton.onClick.AddListener(JoinRoom);

        // connectButton ���� ������ Firebase���� �ٷ� ������ �õ��ϹǷ� �����ϰų� ��Ȱ��ȭ�մϴ�.
        // connectButton.onClick.AddListener(ConnectToServer);

        // ���� �� �κ�/�� �г��� ��Ȱ��ȭ
        lobbyPanel.SetActive(false);
        roomPanel.SetActive(false);
        // connectPanel�� ��Ȱ��ȭ ó��. ���� ���¸� �����ִ� �ؽ�Ʈ ������ ��ü�� �� �ֽ��ϴ�.
        connectPanel.SetActive(true);
    }
    
    // 2. Firebase ID�� ���濡 �����ϴ� public �޼��� (�ٽ�!)
    public void ConnectToPhoton(string userId)
    {
        if (PhotonNetwork.IsConnected)
        {
            Debug.Log("�̹� ������ ����Ǿ� �ֽ��ϴ�.");
            return;
        }

        Debug.Log($"Firebase ID({userId})�� ���� ������ ������ �õ��մϴ�.");
        
        // ���� �߿��� �κ�: Photon�� ���������� Firebase UserId�� ���
        PhotonNetwork.AuthValues = new AuthenticationValues(userId);
        PhotonNetwork.NickName = "Player_" + userId.Substring(0, 5); // �г����� �ӽ÷� ����
        PhotonNetwork.GameVersion = gameVersion;
        PhotonNetwork.ConnectUsingSettings(); // ���� ���� ���� ����
    }

    // ������ ConnectToServer �޼���� ���� ������� �����Ƿ� �����ϰų� �ּ� ó���մϴ�.
    /*
    private void ConnectToServer()
    {
        // ... ���� �ڵ� ...
    }
    */
    
    // �� ����
    private void CreateRoom()
    {
        if (string.IsNullOrEmpty(roomNameInput.text))
        {
            Debug.Log("�� �̸��� �Է��ϼ���.");
            return;
        }
        RoomOptions roomOptions = new RoomOptions();
        roomOptions.MaxPlayers = 4; // �ִ� 4�����
        PhotonNetwork.CreateRoom(roomNameInput.text, roomOptions);
    }
    
    // �� ����
    private void JoinRoom()
    {
        if (string.IsNullOrEmpty(roomNameInput.text))
        {
            Debug.Log("�� �̸��� �Է��ϼ���.");
            return;
        }
        PhotonNetwork.JoinRoom(roomNameInput.text);
    }
    
    #region Photon Callback Methods
    
    // ������ ������ �������� �� ȣ��Ǵ� �ݹ�
    public override void OnConnectedToMaster()
    {
        Debug.Log("������ ������ �����߽��ϴ�.");
        connectPanel.SetActive(false);
        lobbyPanel.SetActive(true); // ���� ���� �� �κ� �г� Ȱ��ȭ
    }

    // �� ������ �������� �� ȣ��Ǵ� �ݹ�
    public override void OnCreatedRoom()
    {
        Debug.Log("�� ���� ����!");
    }

    // �濡 �������� �� ȣ��Ǵ� �ݹ�
    public override void OnJoinedRoom()
    {
        Debug.Log("�濡 �����߽��ϴ�.");
        lobbyPanel.SetActive(false);
        roomPanel.SetActive(true);
        roomNameText.text = "�� �̸�: " + PhotonNetwork.CurrentRoom.Name;
        UpdatePlayerList();
        
        startGameButton.gameObject.SetActive(PhotonNetwork.IsMasterClient);
    }
/*  10.01.12:43 146~151 line ����
    // �ٸ� �÷��̾ �濡 �������� �� ȣ��Ǵ� �ݹ�
    public override void OnPlayerEnteredRoom(Player newPlayer)
    {
        Debug.Log(newPlayer.NickName + "���� �����߽��ϴ�.");
        UpdatePlayerList();
    }
*/
    // �ٸ� �÷��̾ �濡 �������� �� ȣ��Ǵ� �ݹ�
    public override void OnPlayerEnteredRoom(Photon.Realtime.Player newPlayer) // <- Photon.Realtime ���ӽ����̽� ���
    {
        Debug.Log(newPlayer.NickName + "���� �����߽��ϴ�.");
        UpdatePlayerList();
    }
/*  10.01.12:43 161~165 line ����
    // �÷��̾ �濡�� ������ �� ȣ��Ǵ� �ݹ�
    public override void OnPlayerLeftRoom(Player otherPlayer)
    {
        Debug.Log(otherPlayer.NickName + "���� �����߽��ϴ�.");
        UpdatePlayerList();
    }
*/    
    // �÷��̾ �濡�� ������ �� ȣ��Ǵ� �ݹ�
    public override void OnPlayerLeftRoom(Photon.Realtime.Player otherPlayer) // <- Photon.Realtime ���ӽ����̽� ���
    {
        Debug.Log(otherPlayer.NickName + "���� �����߽��ϴ�.");
        UpdatePlayerList();
    }
    #endregion

/*  10.01.12:43 179~188 line ����
    private void UpdatePlayerList()
    {
        string playerListString = "�÷��̾� ���:\n";
        foreach (Player player in PhotonNetwork.PlayerList)
        {
            playerListString += player.NickName + "\n";
        }
        playerListText.text = playerListString;
    }
*/
    private void UpdatePlayerList()
    {
        string playerListString = "�÷��̾� ���:\n";
        // �̰��� Player Ÿ���� ��Ȯ�� �������ݴϴ�.
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