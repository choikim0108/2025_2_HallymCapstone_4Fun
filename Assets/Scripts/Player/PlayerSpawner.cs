using UnityEngine;
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
    }

    // 필요시, 스폰된 플레이어 반환
    public GameObject GetSpawnedPlayer()
    {
        return spawnedPlayer;
    }
}
