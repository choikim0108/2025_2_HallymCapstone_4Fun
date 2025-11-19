using UnityEngine;
using UnityEngine.UI;
using Firebase.Auth;
using Firebase;

public class FirebaseAuthManager : MonoBehaviour
{
    [Header("UI Elements")]
    public InputField emailInput;
    public InputField passwordInput;
    public Button registerButton;
    public Button loginButton;
    public Text statusText;

    private bool isProcessing = false;

    void Start()
    {
        // [수정] 1. 시작할 때 버튼과 입력창을 비활성화
        SetUIInteractable(false);
        statusText.text = "Firebase 초기화 중...";

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
        statusText.text = message;

        if (!success)
        {
            // 실패 시 UI는 계속 비활성화 (로그인 불가)
            SetUIInteractable(false);
            return;
        }

        if (TryRestoreSession())
        {
            return;
        }

        // 유지된 세션이 없다면 수동 입력을 허용
        SetUIInteractable(true);
    }

    // [추가] 4. UI 요소들을 한꺼번에 켜고 끄는 헬퍼 함수
    private void SetUIInteractable(bool interactable)
    {
        emailInput.interactable = interactable;
        passwordInput.interactable = interactable;
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
        statusText.text = $"이미 로그인된 상태입니다: {userLabel}";
        SetUIInteractable(false);

        if (CurrencyManager.Instance != null)
        {
            _ = CurrencyManager.Instance.EnsureInitialDataLoadedAsync(true);
        }

        if (LobbyManager.Instance != null)
        {
            LobbyManager.Instance.ConnectToPhoton(manager.UserId);
        }
        else
        {
            Debug.LogWarning("FirebaseAuthManager: LobbyManager 인스턴스를 찾지 못했습니다. Photon 연결을 건너뜁니다.");
        }

        CurrencyManager.TryPushCachedLoadoutToPhoton();

        return true;
    }

    // [수정] 5. OnRegisterClicked에서 초기화 확인 로직 제거 (이미 검증됨)
    private async void OnRegisterClicked()
    {
        if (isProcessing) return;
        
        // (이전 확인 로직 제거)
        // if (FirebaseManager.Instance == null || !FirebaseManager.Instance.IsFirebaseInitialized) ...

        isProcessing = true;
        statusText.text = "회원가입 중...";

        try
        {
            string email = emailInput.text;
            string password = passwordInput.text;
            
            AuthResult result = await FirebaseManager.Instance.RegisterAsync(email, password);
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

    // [수정] 6. OnLoginClicked에서 초기화 확인 로직 제거 (이미 검증됨)
    private async void OnLoginClicked()
    {
        if (isProcessing) return;
        
        // (이전 확인 로직 제거)

        isProcessing = true;
        statusText.text = "로그인 중...";

        try
        {
            string email = emailInput.text;
            string password = passwordInput.text;
            
            AuthResult result = await FirebaseManager.Instance.LoginAsync(email, password);
            statusText.text = "로그인 성공! 로비에 연결합니다.";

            if (CurrencyManager.Instance != null)
            {
                await CurrencyManager.Instance.EnsureInitialDataLoadedAsync(true);
            }
            
            LobbyManager.Instance.ConnectToPhoton(FirebaseManager.Instance.UserId);
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
}