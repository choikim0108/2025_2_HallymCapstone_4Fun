using UnityEngine;
using UnityEngine.UI;
using Firebase.Auth; // Firebase 인증 네임스페이스
using System.Threading.Tasks; // Task 사용을 위함

public class FirebaseManager : MonoBehaviour
{
    // Firebase 인증 기능의 핵심 객체
    private FirebaseAuth auth;

    [Header("UI Elements")]
    public InputField emailInput;
    public InputField passwordInput;
    public Button registerButton;
    public Button loginButton;
    public Text statusText; // 사용자에게 상태를 알려줄 텍스트 (예: "로그인 중...", "회원가입 성공!")

    void Start()
    {
        // Firebase Auth의 기본 인스턴스를 가져와 초기화
        auth = FirebaseAuth.DefaultInstance;

        // 각 버튼에 클릭했을 때 실행될 함수를 연결
        registerButton.onClick.AddListener(Register);
        loginButton.onClick.AddListener(Login);
    }

    // 회원가입 함수
    private void Register()
    {
        // 사용자가 입력한 이메일과 비밀번호를 가져옴
        string email = emailInput.text;
        string password = passwordInput.text;

        statusText.text = "회원가입 중...";

        // Firebase Auth를 이용해 이메일/비밀번호로 신규 유저 생성
        auth.CreateUserWithEmailAndPasswordAsync(email, password).ContinueWith(task =>
        {
            // task: 비동기 작업의 결과를 담고 있는 객체
            if (task.IsCanceled || task.IsFaulted)
            {
                Debug.LogError("회원가입 실패: " + task.Exception);
                statusText.text = "회원가입에 실패했습니다.";
                return;
            }

            // 회원가입 성공
            FirebaseUser newUser = task.Result.User;
            Debug.LogFormat("회원가입 성공: {0} ({1})", newUser.DisplayName, newUser.UserId);
            statusText.text = "회원가입 성공! 이제 로그인 해주세요.";

        }, TaskScheduler.FromCurrentSynchronizationContext()); // Unity의 메인 스레드에서 UI를 업데이트하기 위함
    }

    // 로그인 함수
    private void Login()
    {
        // 사용자가 입력한 이메일과 비밀번호를 가져옴
        string email = emailInput.text;
        string password = passwordInput.text;

        statusText.text = "로그인 중...";

        // Firebase Auth를 이용해 이메일/비밀번호로 로그인 시도
        auth.SignInWithEmailAndPasswordAsync(email, password).ContinueWith(task =>
        {
            if (task.IsCanceled || task.IsFaulted)
            {
                Debug.LogError("로그인 실패: " + task.Exception);
                statusText.text = "로그인에 실패했습니다.";
                return;
            }

            // 로그인 성공!
            FirebaseUser user = task.Result.User;
            Debug.LogFormat("로그인 성공: {0} ({1})", user.DisplayName, user.UserId);
            statusText.text = "로그인 성공! 로비에 연결합니다.";
            
            // ▼▼▼▼▼ 여기가 가장 중요한 연동 부분 ▼▼▼▼▼
            // LobbyManager의 인스턴스를 찾아 ConnectToPhoton 메서드를 호출하고, 유저의 고유 ID를 넘겨줌
            LobbyManager.Instance.ConnectToPhoton(user.UserId);
            
            // 로그인 UI를 비활성화 (선택 사항)
            // 예를 들어 로그인 패널 전체를 비활성화 할 수 있습니다.
            // this.gameObject.SetActive(false);
            // ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        }, TaskScheduler.FromCurrentSynchronizationContext()); // Unity의 메인 스레드에서 UI를 업데이트하기 위함
    }
}