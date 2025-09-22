using UnityEngine;

public class PlayerSpawner : MonoBehaviour
{
    public GameObject playerPrefab;
 
    public Transform spawnPoint;

    private GameObject spawnedPlayer;

    void Start()
    {
        Vector3 spawnPos = spawnPoint ? spawnPoint.position : transform.position;
        Quaternion spawnRot = spawnPoint ? spawnPoint.rotation : transform.rotation;
        spawnedPlayer = Instantiate(playerPrefab, spawnPos, spawnRot);
    }

    // 필요시, 스폰된 플레이어 반환
    public GameObject GetSpawnedPlayer()
    {
        return spawnedPlayer;
    }
}
