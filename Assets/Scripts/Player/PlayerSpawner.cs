using UnityEngine;
using System.Collections.Generic;
using Photon.Pun;

public class PlayerSpawner : MonoBehaviour
{
    public GameObject playerPrefab;

    public Transform spawnPoint;

    private GameObject spawnedPlayer;

    void Start()
    {
        Vector3 spawnPos = spawnPoint ? spawnPoint.position : transform.position;
        Quaternion spawnRot = spawnPoint ? spawnPoint.rotation : transform.rotation;
        Debug.Log($"[PlayerSpawner] PhotonNetwork.InRoom: {Photon.Pun.PhotonNetwork.InRoom}");
        Debug.Log($"[PlayerSpawner] playerPrefab.name: {playerPrefab.name}");
        spawnedPlayer = Photon.Pun.PhotonNetwork.Instantiate(playerPrefab.name, spawnPos, spawnRot);
        Debug.Log($"[PlayerSpawner] spawnedPlayer: {spawnedPlayer}");

        PlayerLoadoutApplier applier = spawnedPlayer.GetComponentInChildren<PlayerLoadoutApplier>();
        if (applier != null)
        {
            Dictionary<string, string> loadout = GetLocalLoadout();
            int count = loadout != null ? loadout.Count : 0;
            Debug.Log($"[PlayerSpawner] Applying local loadout to spawned player. itemCount={count}");
            applier.ApplyLocalLoadout(loadout);
        }
    }

    private static Dictionary<string, string> GetLocalLoadout()
    {
        if (CurrencyManager.Instance != null)
        {
            return CurrencyManager.Instance.GetCurrentLoadoutSnapshot();
        }

        return CurrencyManager.GetLastKnownLoadoutSnapshot();
    }

    // 필요시, 스폰된 플레이어 반환
    public GameObject GetSpawnedPlayer()
    {
        return spawnedPlayer;
    }
}
