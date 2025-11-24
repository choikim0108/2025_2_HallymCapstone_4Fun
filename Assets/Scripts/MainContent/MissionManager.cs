using UnityEngine;
using Firebase.Firestore;
using System.Collections.Generic;
using System.Threading.Tasks;

public class MissionManager : MonoBehaviour
{
    public static MissionManager Instance { get; private set; }

    [Header("Rewards")]
    [SerializeField] private long missionReward = 50;

    [Header("Audio")]
    [SerializeField] private AudioClip missionRewardSound;
    private AudioSource _audioSource;

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Debug.LogWarning("[MissionManager] Duplicate instance detected. Destroying the newer instance.");
            Destroy(gameObject);
            return;
        }

        Instance = this;

        _audioSource = gameObject.AddComponent<AudioSource>();
        _audioSource.playOnAwake = false;

        if (missionRewardSound == null)
        {
            missionRewardSound = Resources.Load<AudioClip>("Sounds/MissionReward");
        }
    }

    public void MissionFail(string reason)
    {
        Debug.Log($"[MissionManager] Mission Failed: {reason}");
    }

    public async void MissionSuccess(string missionId = null)
    {
        if (string.IsNullOrEmpty(missionId))
        {
            Debug.Log("[MissionManager] Mission Success");
        }
        else
        {
            Debug.Log($"[MissionManager] Mission Success: {missionId}");
        }

        await AddCurrencyRewardAsync();

        if (missionRewardSound != null && _audioSource != null)
        {
            _audioSource.PlayOneShot(missionRewardSound);
        }
    }

    private async Task AddCurrencyRewardAsync()
    {
        CurrencyManager currencyManager = CurrencyManager.Instance;
        if (currencyManager == null)
        {
            Debug.LogWarning("[MissionManager] CurrencyManager not found. Cannot add reward.");
            return;
        }

        FirebaseFirestore db = currencyManager.Database;
        string userId = currencyManager.UserId;

        if (db == null || string.IsNullOrWhiteSpace(userId))
        {
            Debug.LogWarning("[MissionManager] Firebase not initialized. Cannot add reward.");
            return;
        }

        try
        {
            DocumentReference userDocRef = db.Collection("users").Document(userId);

            await db.RunTransactionAsync(async transaction =>
            {
                DocumentSnapshot snapshot = await transaction.GetSnapshotAsync(userDocRef);
                long currentCurrency = snapshot.ContainsField("currency")
                    ? snapshot.GetValue<long>("currency")
                    : 0L;

                long updatedCurrency = currentCurrency + missionReward;

                transaction.Update(userDocRef, new Dictionary<string, object>
                {
                    { "currency", updatedCurrency }
                });

                return updatedCurrency;
            });

            Debug.Log($"[MissionManager] Added {missionReward} currency as mission reward.");
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[MissionManager] Failed to add currency reward: {ex.Message}");
        }
    }
}
