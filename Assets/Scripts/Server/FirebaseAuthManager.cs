using UnityEngine;
using UnityEngine.UI;
using Firebase.Auth;
using Firebase;
using TMPro;


public class FirebaseAuthManager : MonoBehaviour
{
    [Header("UI Elements")]
    public TMP_InputField emailInput;
    public TMP_InputField passwordInput;
    public TMP_InputField nicknameInput;   // ★ 추가: 닉네임 입력칸
    public Button registerButton;
    public Button loginButton;
    public TextMeshProUGUI statusText;     // ★ Text → TextMeshProUGUI 로 변경

    private bool isProcessing = false;

    void Start()
    {
        // [수정] 1. 시작할 때 버튼과 입력창을 비활성화
        SetUIInteractable(false);
        statusText.text = "";
        Debug.Log("Firebase 초기화 중..."); // Firebase 초기화 메시지 콘솔에만 표기(화면X)

        // [수정] 2. FirebaseManager가 준비 완료 이벤트를 보낼 때까지 대기
        // (싱글톤 인스턴스는 Awake에서 생성되므로 Start에서 접근해도 안전)
        if (FirebaseManager.Instance != null)
        {
            FirebaseManager.Instance.OnFirebaseInitialized += HandleFirebaseReady;

            if (FirebaseManager.Instance.IsFirebaseInitialized)
            {
                // 이미 초기화가 끝난 상태로 씬에 진입한 경우 즉시 처리
                HandleFirebaseReady(true, "Firebase 준비 완료");
            }
        }
        else
        {
            // 이 경우는 FirebaseManager가 씬에 아예 없는 경우
            statusText.text = "FirebaseManager를 찾을 수 없습니다!";
            Debug.LogError("FirebaseManager.Instance가 null입니다. 씬에 FirebaseManager가 있는지 확인하세요.");
        }

        // 기존 리스너 연결은 유지
        registerButton.onClick.AddListener(OnRegisterClicked);
        loginButton.onClick.AddListener(OnLoginClicked);
    }

    //TAB 키 입력을 감지하기 위해 Update 추가
    void Update()
    {
        // TAB 키가 눌렸는지 확인
        if (Input.GetKeyDown(KeyCode.Tab))
        {
            // 현재 포커스된 InputField에 따라 다음 InputField로 포커스 이동
            if (emailInput.isFocused)
            {
                passwordInput.ActivateInputField(); // 이메일 -> 비밀번호
            }
            else if (passwordInput.isFocused)
            {
                nicknameInput.ActivateInputField(); // 비밀번호 -> 닉네임
            }
            else if (nicknameInput.isFocused)
            {
                emailInput.ActivateInputField(); // 닉네임 -> 이메일
            }
        }
        // 엔터키(Return) 입력 시 로그인 시도
        // 일반 엔터(Return) 또는 키패드 엔터(KeypadEnter) 감지
        if (Input.GetKeyDown(KeyCode.Return) || Input.GetKeyDown(KeyCode.KeypadEnter))
        {
            // 현재 로직이 처리 중이 아니고, 로그인 버튼이 활성화된 상태라면 로그인 시도
            if (!isProcessing && loginButton.interactable)
            {
                OnLoginClicked();
            }
        }
    }
    private void OnDestroy()
    {
        if (FirebaseManager.Instance != null)
        {
            FirebaseManager.Instance.OnFirebaseInitialized -= HandleFirebaseReady;
        }
    }

    // [추가] 3. FirebaseManager로부터 초기화 완료 신호를 받는 함수
    private void HandleFirebaseReady(bool success, string message)
    {
        if (!success)
        {
            // 실패 메시지는 화면에 그대로 표시
            statusText.text = message;
            SetUIInteractable(false);
            return;
        }

        // 성공 메시지는 콘솔만 출력, 화면에는 표시 X
        Debug.Log(message);
        statusText.text = "";   // 화면 초기화

        if (TryRestoreSession())
        {
            return;
        }

        SetUIInteractable(true);
    }

    // [추가] 4. UI 요소들을 한꺼번에 켜고 끄는 헬퍼 함수
    private void SetUIInteractable(bool interactable)
    {
        emailInput.interactable = interactable;
        passwordInput.interactable = interactable;
        nicknameInput.interactable = interactable;
        registerButton.interactable = interactable;
        loginButton.interactable = interactable;
    }

    private bool TryRestoreSession()
    {
        FirebaseManager manager = FirebaseManager.Instance;
        if (manager == null || !manager.HasCachedUser)
        {
            return false;
        }

        FirebaseUser currentUser = manager.User;
        if (currentUser == null && manager.Auth != null)
        {
            currentUser = manager.Auth.CurrentUser;
        }

        if (currentUser == null)
        {
            return false;
        }

        manager.CacheUser(currentUser);

        string userLabel = string.IsNullOrWhiteSpace(currentUser.Email) ? manager.UserId : currentUser.Email;
        Debug.Log($"이미 로그인된 상태: {userLabel}");
        // 수정) UI에 텍스트는 표시하지 않음
        statusText.text = "";
        SetUIInteractable(false);

        if (CurrencyManager.Instance != null)
        {
            _ = CurrencyManager.Instance.EnsureInitialDataLoadedAsync(true);
        }

        if (LobbyManager.Instance != null)
        {
            string cachedNickname = "Player_" + manager.UserId.Substring(0, 5);
            LobbyManager.Instance.ConnectToPhoton(manager.UserId, cachedNickname);        
        }
        else
        {
            Debug.LogWarning("FirebaseAuthManager: LobbyManager 인스턴스를 찾지 못했습니다. Photon 연결을 건너뜁니다.");
        }

        CurrencyManager.TryPushCachedLoadoutToPhoton();

        return true;
    }

    //OnRegisterClicked에서 초기화 확인 로직 제거 (이미 검증됨)
    private async void OnRegisterClicked()
    {
        if (isProcessing) return;
        
        if (string.IsNullOrEmpty(nicknameInput.text))
        {
            statusText.text = "닉네임을 입력하세요."; // 요구조건 2
            return;
        }

        isProcessing = true;
        statusText.text = "회원가입 중...";

        try
        {
            string email = emailInput.text;
            string password = passwordInput.text;
            string nickname = nicknameInput.text;
            
            AuthResult result = await FirebaseManager.Instance.RegisterAsync(email, password, nickname);
            statusText.text = "회원가입 성공! 이제 로그인 해주세요.";
        }
        catch (System.Exception e)
        {
            HandleAuthError(e, "회원가입");
        }
        finally
        {
            isProcessing = false;
        }
    }

    // OnLoginClicked에서 초기화 확인 로직 제거 (이미 검증됨)
    private async void OnLoginClicked()
    {
        if (isProcessing) return;
        
        // (이전 확인 로직 제거)
        // 요구조건 1: 닉네임이 비어있는지 확인
        if (string.IsNullOrEmpty(nicknameInput.text))
        {
            // 요구조건 2: 닉네임이 비어있을 경우 상태 메시지 변경
            statusText.text = "닉네임을 입력하세요.";
            return;
        }
        isProcessing = true;
        statusText.text = "로그인 중...";

        try
        {
            string email = emailInput.text;
            string password = passwordInput.text;
            string nickname = nicknameInput.text; // 닉네임 값 가져오기
            
            AuthResult result = await FirebaseManager.Instance.LoginAsync(email, password);
            statusText.text = "로그인 성공! 로비에 연결합니다.";

            if (CurrencyManager.Instance != null)
            {
                await CurrencyManager.Instance.EnsureInitialDataLoadedAsync(true);
            }
            
            LobbyManager.Instance.ConnectToPhoton(FirebaseManager.Instance.UserId, nickname);        
        }
        catch (System.Exception e)
        {
            HandleAuthError(e, "로그인");
        }
        finally
        {
            isProcessing = false;
        }
    }
    
    // (HandleAuthError, GetAuthErrorMessage 함수는 기존과 동일)
    private void HandleAuthError(System.Exception e, string operationType)
    {
        Debug.LogError($"{operationType} 실패: {e.Message}");

        if (e.GetBaseException() is FirebaseException firebaseEx)
        {
            AuthError errorCode = (AuthError)firebaseEx.ErrorCode;
            statusText.text = GetAuthErrorMessage(errorCode);
        }
        else
        {
            statusText.text = $"{operationType}에 실패했습니다.";
        }
    }

    private string GetAuthErrorMessage(AuthError errorCode)
    {
        switch (errorCode)
        {
            case AuthError.WrongPassword:
                return "비밀번호가 틀렸습니다.";
            case AuthError.UserNotFound:
                return "존재하지 않는 계정입니다.";
            case AuthError.InvalidEmail:
                return "유효하지 않은 이메일 주소입니다.";
            case AuthError.EmailAlreadyInUse:
                return "이미 사용 중인 이메일입니다.";
            case AuthError.WeakPassword:
                return "비밀번호는 6자리 이상이어야 합니다.";
            default:
                return "오류가 발생했습니다: " + errorCode.ToString();
        }
    }

    public void ClearStatusText() /* 로그인 성공 후 로비패널 넘어가서 "로그인 성공!" 텍스트 삭제를 위한 '지우는 함수' */
    {
        if (statusText != null)
            statusText.text = "";
    }

}