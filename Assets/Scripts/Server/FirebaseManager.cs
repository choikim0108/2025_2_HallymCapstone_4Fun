using UnityEngine;
using Firebase;
using Firebase.Auth;
using Firebase.Firestore;
using System.Threading.Tasks;
using System.Collections.Generic;
using System; //System.Action 호출용
using System.IO;

public class FirebaseManager : MonoBehaviour
{
    public static FirebaseManager Instance { get; private set; }

    public FirebaseApp App { get; private set; }
    public FirebaseAuth Auth { get; private set; }
    public FirebaseFirestore DB { get; private set; }

    public FirebaseUser User { get; private set; }
    public string UserId { get; private set; }
    
    public bool IsFirebaseInitialized { get; private set; } = false;
    public bool HasCachedUser { get; private set; } = false;

    // [추가] 1. Firebase 초기화 상태를 알릴 C# 이벤트
    public event Action<bool, string> OnFirebaseInitialized; // <성공여부, 메시지>

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
            InitializeFirebase();
        }
        else
        {
            Destroy(gameObject);
        }
    }

    private void InitializeFirebase()
    {
        FirebaseApp.CheckAndFixDependenciesAsync().ContinueWith(task =>
        {
            var dependencyStatus = task.Result;
            if (dependencyStatus == DependencyStatus.Available)
            {
                // Firebase가 준비됨
                App = FirebaseApp.DefaultInstance;
                Auth = FirebaseAuth.DefaultInstance;
                var firestore = FirebaseFirestore.DefaultInstance;
#if UNITY_EDITOR
                try
                {
                    var projectRoot = Application.dataPath.Replace("/Assets", string.Empty);
                    var cloneMarkerPath = Path.Combine(projectRoot, ".clone");
                    if (File.Exists(cloneMarkerPath))
                    {
                        firestore.Settings.PersistenceEnabled = false;
                        Debug.Log("Firestore persistence disabled for ParrelSync clone instance.");
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"Clone detection for Firestore settings failed: {ex.Message}");
                }
#endif
                DB = firestore;

                IsFirebaseInitialized = true;
                Debug.Log("Firebase 초기화 성공!");
                
                // [수정] 메인 스레드에서 실행되므로 바로 이벤트 호출
                OnFirebaseInitialized?.Invoke(true, "Firebase 준비 완료");
            }
            else
            {
                IsFirebaseInitialized = false;
                string message = $"Firebase 초기화 실패: {dependencyStatus}";
                Debug.LogError(message);
                
                // [수정] 메인 스레드에서 실행되므로 바로 이벤트 호출
                OnFirebaseInitialized?.Invoke(false, message);
            }
        }, TaskScheduler.FromCurrentSynchronizationContext()); // 핵심
    }

    public async Task<AuthResult> RegisterAsync(string email, string password)
    {
        if (!IsFirebaseInitialized) throw new System.Exception("Firebase not ready.");

        // 1. Auth로 유저 생성
        AuthResult authResult = await Auth.CreateUserWithEmailAndPasswordAsync(email, password);
        FirebaseUser newUser = authResult.User;
        Debug.LogFormat("Auth 회원가입 성공: {0} ({1})", newUser.DisplayName, newUser.UserId);

        // 2. "배치(Batch) 작업"을 생성합니다.
        WriteBatch batch = DB.StartBatch();

        // 3-1. (쓰기 1) 기본 유저 문서 참조
        DocumentReference userDocRef = DB.Collection("users").Document(newUser.UserId);
        var newPlayerData = new Dictionary<string, object>
        {
            { "nickname", "NewPlayer" + UnityEngine.Random.Range(100, 999) },
            { "email", email },
            { "currency", 100 }, // 기본 재화
            { "created_at", FieldValue.ServerTimestamp },
            
            // 3-2. 기본 꾸미기 정보(Loadout)
            { "current_loadout", new Dictionary<string, object> {
                { "shirt", "default_shirt" },
                { "pants", "default_pants" },
                { "face", "default_face" }
            }}
        };
        batch.Set(userDocRef, newPlayerData); 

        // 3-3. (쓰기 2) 'inventory' 하위 컬렉션에 기본 셔츠 아이템 추가
        DocumentReference shirtRef = userDocRef.Collection("inventory").Document("default_shirt");
        var shirtData = new Dictionary<string, object> { { "acquired_at", FieldValue.ServerTimestamp } };
        batch.Set(shirtRef, shirtData); 

        // 3-4. (쓰기 3) 'inventory' 하위 컬렉션에 기본 바지 아이템 추가
        DocumentReference pantsRef = userDocRef.Collection("inventory").Document("default_pants");
        var pantsData = new Dictionary<string, object> { { "acquired_at", FieldValue.ServerTimestamp } };
        batch.Set(pantsRef, pantsData); 

        // 3-5. (쓰기 4) 'inventory' 하위 컬렉션에 기본 얼굴 아이템 추가
        DocumentReference faceRef = userDocRef.Collection("inventory").Document("default_face");
        var faceData = new Dictionary<string, object> { { "acquired_at", FieldValue.ServerTimestamp } };
        batch.Set(faceRef, faceData);

        // 4. (실행) 묶어둔 모든 쓰기 작업을 "전부 성공" 또는 "전부 실패"로 서버에 전송
        await batch.CommitAsync(); 
        
        Debug.Log($"Firestore에 {newUser.UserId} 유저 데이터 및 기본 아이템 생성 완료!");
        
        return authResult;
    }

    public async Task<AuthResult> LoginAsync(string email, string password)
    {
        if (!IsFirebaseInitialized) throw new System.Exception("Firebase not ready.");
        
        AuthResult authResult = await Auth.SignInWithEmailAndPasswordAsync(email, password);
        
        CacheUser(authResult.User);
        Debug.LogFormat("Auth 로그인 성공: {0} ({1})", User.DisplayName, UserId);
        
        return authResult;
    }

    public void Logout()
    {
        if (Auth != null)
        {
            Auth.SignOut();
        }
        User = null;
        UserId = null;
        HasCachedUser = false;
    }

    public void CacheUser(FirebaseUser firebaseUser)
    {
        if (firebaseUser == null)
        {
            User = null;
            UserId = null;
            HasCachedUser = false;
            return;
        }

        User = firebaseUser;
        UserId = firebaseUser.UserId;
        HasCachedUser = true;
    }
}