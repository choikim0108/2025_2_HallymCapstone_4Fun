using System;
using System.Collections.Generic;
using UnityEngine;

public static class LoadoutSerializer
{
    [Serializable]
    private class SerializableLoadout
    {
        public List<string> keys = new();
        public List<string> values = new();
    }

    public static string Serialize(IReadOnlyDictionary<string, string> loadout)
    {
        SerializableLoadout container = new SerializableLoadout();

        if (loadout != null)
        {
            foreach (KeyValuePair<string, string> entry in loadout)
            {
                if (string.IsNullOrWhiteSpace(entry.Key) || string.IsNullOrWhiteSpace(entry.Value))
                {
                    continue;
                }

                container.keys.Add(entry.Key);
                container.values.Add(entry.Value);
            }
        }

        return JsonUtility.ToJson(container);
    }

    public static Dictionary<string, string> Deserialize(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return null;
        }

        SerializableLoadout container = JsonUtility.FromJson<SerializableLoadout>(json);
        if (container == null)
        {
            return null;
        }

        Dictionary<string, string> result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (container.keys == null || container.values == null)
        {
            return result;
        }

        int count = Mathf.Min(container.keys.Count, container.values.Count);
        for (int i = 0; i < count; i++)
        {
            string key = container.keys[i];
            string value = container.values[i];
            if (string.IsNullOrWhiteSpace(key) || string.IsNullOrWhiteSpace(value))
            {
                continue;
            }

            result[key] = value;
        }

        return result;
    }
}
