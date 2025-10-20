using UnityEngine;

public class MissionManager : MonoBehaviour
{
    public static MissionManager Instance { get; private set; }

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Debug.LogWarning("[MissionManager] Duplicate instance detected. Destroying the newer instance.");
            Destroy(gameObject);
            return;
        }

        Instance = this;
    }

    public void MissionFail(string reason)
    {
        Debug.Log($"[MissionManager] Mission Failed: {reason}");
        // Additional fail handling (UI/audio/network sync) can be added here.
    }

    public void MissionSuccess(string missionId = null)
    {
        if (string.IsNullOrEmpty(missionId))
        {
            Debug.Log("[MissionManager] Mission Success");
        }
        else
        {
            Debug.Log($"[MissionManager] Mission Success: {missionId}");
        }
        // TODO: Integrate Firebase DB updates for rewards/achievements when backend is ready.
    }
}
