using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Firebase.Firestore;
using UnityEngine;

public static class CurrencyManagerLoader
{
    public static async Task<Dictionary<string, string>> FetchLoadoutAsync(FirebaseFirestore db, string userId)
    {
        if (db == null || string.IsNullOrWhiteSpace(userId))
        {
            Debug.LogWarning("CurrencyManagerLoader.FetchLoadoutAsync: invalid db or userId");
            return null;
        }

        try
        {
            DocumentReference docRef = db.Collection("users").Document(userId);
            DocumentSnapshot snapshot = await docRef.GetSnapshotAsync();
            if (!snapshot.Exists)
            {
                Debug.LogWarning("CurrencyManagerLoader: user document missing");
                return null;
            }

            Dictionary<string, object> rawLoadout = null;
            if (snapshot.TryGetValue("current_loadout", out Dictionary<string, object> loadoutMap))
            {
                rawLoadout = loadoutMap;
            }

            Dictionary<string, string> normalized = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            if (rawLoadout != null)
            {
                foreach (KeyValuePair<string, object> pair in rawLoadout)
                {
                    if (!string.IsNullOrWhiteSpace(pair.Key) && pair.Value is string value && !string.IsNullOrWhiteSpace(value))
                    {
                        normalized[pair.Key] = value;
                    }
                }
            }

            return normalized;
        }
        catch (Exception ex)
        {
            Debug.LogError($"CurrencyManagerLoader.FetchLoadoutAsync Error: {ex.Message}");
            return null;
        }
    }
}