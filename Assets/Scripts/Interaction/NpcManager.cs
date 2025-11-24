using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class NpcManager : MonoBehaviour
{
    public static NpcManager Instance;

    [Header("Settings")]
    public Transform spawnPointRoot;
    public string npcResourcePath = "NPCs";

    private List<Transform> spawnPoints = new List<Transform>();
    private GameObject[] npcPrefabs;
    
    // Track spawned NPCs: SpawnPoint -> NPC GameObject
    private Dictionary<Transform, GameObject> spawnedNpcs = new Dictionary<Transform, GameObject>();
    
    // Track previous spawn to avoid duplicates on same spot: SpawnPoint -> Prefab Name
    private Dictionary<Transform, string> lastSpawnedPrefabNames = new Dictionary<Transform, string>();

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
        }
    }

    void Start()
    {
        Initialize();
        SpawnNpcs();
    }

    private void Initialize()
    {
        // Load NPC Prefabs
        npcPrefabs = Resources.LoadAll<GameObject>(npcResourcePath);
        if (npcPrefabs == null || npcPrefabs.Length == 0)
        {
            Debug.LogError($"[NpcManager] No NPC prefabs found at Resources/{npcResourcePath}");
        }

        // Find Spawn Points
        if (spawnPointRoot == null)
        {
            GameObject rootObj = GameObject.Find("NpcSpawnPoints");
            if (rootObj != null)
            {
                spawnPointRoot = rootObj.transform;
            }
        }

        if (spawnPointRoot != null)
        {
            spawnPoints.Clear();
            foreach (Transform child in spawnPointRoot)
            {
                spawnPoints.Add(child);
            }
        }
        else
        {
            Debug.LogError("[NpcManager] NpcSpawnPoints root not found!");
        }
    }

    public void ShuffleNpcs()
    {
        // Destroy existing NPCs
        foreach (var kvp in spawnedNpcs)
        {
            if (kvp.Value != null)
            {
                Destroy(kvp.Value);
            }
        }
        spawnedNpcs.Clear();

        // Respawn with avoidance logic
        SpawnNpcs(true);
    }

    private void SpawnNpcs(bool avoidPrevious = false)
    {
        if (npcPrefabs == null || npcPrefabs.Length == 0) return;
        if (spawnPoints.Count == 0) return;

        List<GameObject> availableNpcs = npcPrefabs.ToList();
        List<Transform> availablePoints = new List<Transform>(spawnPoints);

        // We want to assign NPCs to Points.
        // Strategy:
        // 1. Shuffle Points (to decide which points get filled if Points > NPCs)
        // 2. Shuffle NPCs (to decide which NPCs are used if NPCs > Points, and their order)
        // 3. Try to minimize collisions with lastSpawnedPrefabNames

        int maxAttempts = 20;
        List<GameObject> bestNpcOrder = null;
        List<Transform> bestPointOrder = null;
        int minCollisions = int.MaxValue;

        // Determine how many we can spawn
        int spawnCount = Mathf.Min(availablePoints.Count, availableNpcs.Count);

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            // Shuffle both lists
            var shuffledNpcs = availableNpcs.OrderBy(x => Random.value).ToList();
            var shuffledPoints = availablePoints.OrderBy(x => Random.value).ToList();

            int collisions = 0;
            if (avoidPrevious)
            {
                for (int i = 0; i < spawnCount; i++)
                {
                    Transform pt = shuffledPoints[i];
                    GameObject npc = shuffledNpcs[i];
                    
                    if (lastSpawnedPrefabNames.ContainsKey(pt) && lastSpawnedPrefabNames[pt] == npc.name)
                    {
                        collisions++;
                    }
                }
            }

            if (collisions < minCollisions)
            {
                minCollisions = collisions;
                bestNpcOrder = shuffledNpcs;
                bestPointOrder = shuffledPoints;

                if (collisions == 0) break; // Found a perfect shuffle
            }
        }

        // Apply the best spawn configuration
        Dictionary<Transform, string> newSpawnRecord = new Dictionary<Transform, string>();

        for (int i = 0; i < spawnCount; i++)
        {
            Transform pt = bestPointOrder[i];
            GameObject prefab = bestNpcOrder[i];

            // Spawn
            GameObject newNpc = Instantiate(prefab, pt.position, pt.rotation);
            newNpc.name = prefab.name; // Remove "(Clone)" suffix
            // Optionally parent to the spawn point to keep hierarchy clean, or keep root. 
            // User said "child of cubes" in the context of finding them, but for spawning "at the position".
            // I'll parent them to the spawn point for tidiness.
            newNpc.transform.SetParent(pt); 

            spawnedNpcs[pt] = newNpc;
            newSpawnRecord[pt] = prefab.name;
        }

        // Update history
        lastSpawnedPrefabNames = newSpawnRecord;
        
        Debug.Log($"[NpcManager] Spawned {spawnCount} NPCs. Collisions with previous: {minCollisions}");
    }
}
