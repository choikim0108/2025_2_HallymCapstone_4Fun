using UnityEngine;
using UnityEngine.UI;
using Firebase.Auth; // Firebase ���� ���ӽ����̽�
using System.Threading.Tasks; // Task ����� ����

public class FirebaseManager : MonoBehaviour
{
    // Firebase ���� ����� �ٽ� ��ü
    private FirebaseAuth auth;

    [Header("UI Elements")]
    public InputField emailInput;
    public InputField passwordInput;
    public Button registerButton;
    public Button loginButton;
    public Text statusText; // ����ڿ��� ���¸� �˷��� �ؽ�Ʈ (��: "�α��� ��...", "ȸ������ ����!")

    void Start()
    {
        // Firebase Auth�� �⺻ �ν��Ͻ��� ������ �ʱ�ȭ
        auth = FirebaseAuth.DefaultInstance;

        // �� ��ư�� Ŭ������ �� ����� �Լ��� ����
        registerButton.onClick.AddListener(Register);
        loginButton.onClick.AddListener(Login);
    }

    // ȸ������ �Լ�
    private void Register()
    {
        // ����ڰ� �Է��� �̸��ϰ� ��й�ȣ�� ������
        string email = emailInput.text;
        string password = passwordInput.text;

        statusText.text = "ȸ������ ��...";

        // Firebase Auth�� �̿��� �̸���/��й�ȣ�� �ű� ���� ����
        auth.CreateUserWithEmailAndPasswordAsync(email, password).ContinueWith(task =>
        {
            // task: �񵿱� �۾��� ����� ��� �ִ� ��ü
            if (task.IsCanceled || task.IsFaulted)
            {
                Debug.LogError("ȸ������ ����: " + task.Exception);
                statusText.text = "ȸ�����Կ� �����߽��ϴ�.";
                return;
            }

            // ȸ������ ����
            FirebaseUser newUser = task.Result.User;
            Debug.LogFormat("ȸ������ ����: {0} ({1})", newUser.DisplayName, newUser.UserId);
            statusText.text = "ȸ������ ����! ���� �α��� ���ּ���.";

        }, TaskScheduler.FromCurrentSynchronizationContext()); // Unity�� ���� �����忡�� UI�� ������Ʈ�ϱ� ����
    }

    // �α��� �Լ�
    private void Login()
    {
        // ����ڰ� �Է��� �̸��ϰ� ��й�ȣ�� ������
        string email = emailInput.text;
        string password = passwordInput.text;

        statusText.text = "�α��� ��...";

        // Firebase Auth�� �̿��� �̸���/��й�ȣ�� �α��� �õ�
        auth.SignInWithEmailAndPasswordAsync(email, password).ContinueWith(task =>
        {
            if (task.IsCanceled || task.IsFaulted)
            {
                Debug.LogError("�α��� ����: " + task.Exception);
                statusText.text = "�α��ο� �����߽��ϴ�.";
                return;
            }

            // �α��� ����!
            FirebaseUser user = task.Result.User;
            Debug.LogFormat("�α��� ����: {0} ({1})", user.DisplayName, user.UserId);
            statusText.text = "�α��� ����! �κ� �����մϴ�.";
            
            // ������ ���Ⱑ ���� �߿��� ���� �κ� ������
            // LobbyManager�� �ν��Ͻ��� ã�� ConnectToPhoton �޼��带 ȣ���ϰ�, ������ ���� ID�� �Ѱ���
            LobbyManager.Instance.ConnectToPhoton(user.UserId);
            
            // �α��� UI�� ��Ȱ��ȭ (���� ����)
            // ���� ��� �α��� �г� ��ü�� ��Ȱ��ȭ �� �� �ֽ��ϴ�.
            // this.gameObject.SetActive(false);
            // ����������������������������������

        }, TaskScheduler.FromCurrentSynchronizationContext()); // Unity�� ���� �����忡�� UI�� ������Ʈ�ϱ� ����
    }
}