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

// 회원가입 시 닉네임도 함께 받도록 매개변수 추가
    public async Task<AuthResult> RegisterAsync(string email, string password, string nickname)
    {
        if (!IsFirebaseInitialized) throw new System.Exception("Firebase not ready.");

        AuthResult authResult = await Auth.CreateUserWithEmailAndPasswordAsync(email, password);
        FirebaseUser newUser = authResult.User;
        Debug.LogFormat("Auth 회원가입 성공: {0} ({1})", newUser.DisplayName, newUser.UserId);

        WriteBatch batch = DB.StartBatch();

        DocumentReference userDocRef = DB.Collection("users").Document(newUser.UserId);
        var newPlayerData = new Dictionary<string, object>
        {
            { "nickname", nickname }, 
            { "email", email },
            { "currency", 100 },
            { "created_at", FieldValue.ServerTimestamp },
            
            { "current_loadout", new Dictionary<string, object> {
                { "shirt", "default_shirt" },
                { "pants", "default_pants" },
                { "face", "default_face" }
            }}
        };
        batch.Set(userDocRef, newPlayerData); 

        DocumentReference shirtRef = userDocRef.Collection("inventory").Document("default_shirt");
        var shirtData = new Dictionary<string, object> { { "acquired_at", FieldValue.ServerTimestamp } };
        batch.Set(shirtRef, shirtData); 

        DocumentReference pantsRef = userDocRef.Collection("inventory").Document("default_pants");
        var pantsData = new Dictionary<string, object> { { "acquired_at", FieldValue.ServerTimestamp } };
        batch.Set(pantsRef, pantsData); 

        DocumentReference faceRef = userDocRef.Collection("inventory").Document("default_face");
        var faceData = new Dictionary<string, object> { { "acquired_at", FieldValue.ServerTimestamp } };
        batch.Set(faceRef, faceData);

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