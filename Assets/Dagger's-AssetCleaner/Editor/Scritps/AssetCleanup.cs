using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace Dagger.AssetCleanupTool
{
    public static class AssetCleanup
    {
        public static List<string> ScanUnusedAssets(List<SceneAsset> selectedScenes, List<DefaultAsset> selectedFolders)
        {
            var unusedAssets = new List<string>();
            var usedAssets = new HashSet<string>();

            foreach (var sceneAsset in selectedScenes)
            {
                if (sceneAsset == null) continue;
                string scenePath = AssetDatabase.GetAssetPath(sceneAsset);
                var dependencies = AssetDatabase.GetDependencies(scenePath, true);
                foreach (var dep in dependencies)
                    usedAssets.Add(dep);
            }

            string[] resourceAssets = AssetDatabase.FindAssets("", new[] { "Assets/Resources" });
            foreach (var guid in resourceAssets)
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                usedAssets.Add(path);
            }

            string selfPath = AssetDatabase.GetAssetPath(MonoScript.FromScriptableObject(ScriptableObject.CreateInstance<AssetCleanupToolEditor>()));

            foreach (var folder in selectedFolders)
            {
                if (folder == null) continue;
                string folderPath = AssetDatabase.GetAssetPath(folder);
                string[] guids = AssetDatabase.FindAssets("", new[] { folderPath });

                foreach (string guid in guids)
                {
                    string path = AssetDatabase.GUIDToAssetPath(guid);
                    if (Directory.Exists(path)) continue;
                    if (usedAssets.Contains(path)) continue;
                    if (path.EndsWith(".cs") || path.EndsWith(".unity") || path.EndsWith(".dll")) continue;
                    if (path == selfPath) continue;

                    unusedAssets.Add(path);
                }
            }

            return unusedAssets;
        }

        public static List<string> FilterAssets(
            List<string> assets,
            string search,
            bool textures,
            bool models,
            bool audio,
            bool materials,
            bool shaders,
            bool prefabs,
            bool others)
        {
            return assets.Where(asset =>
            {
                string lower = asset.ToLower();

                if (!string.IsNullOrEmpty(search) && !lower.Contains(search.ToLower()))
                    return false;

                if (lower.EndsWith(".png") || lower.EndsWith(".jpg") || lower.EndsWith(".jpeg") || lower.EndsWith(".tga"))
                    return textures;
                if (lower.EndsWith(".fbx") || lower.EndsWith(".obj"))
                    return models;
                if (lower.EndsWith(".mp3") || lower.EndsWith(".wav") || lower.EndsWith(".ogg"))
                    return audio;
                if (lower.EndsWith(".mat"))
                    return materials;
                if (lower.EndsWith(".shader") || lower.EndsWith(".compute"))
                    return shaders;
                if (lower.EndsWith(".prefab"))
                    return prefabs;

                return others;
            }).ToList();
        }

        public static void DeleteAssets(IEnumerable<string> assetPaths)
        {
            foreach (var path in assetPaths)
            {
                if (!AssetDatabase.DeleteAsset(path))
                    Debug.LogWarning($"Failed to delete: {path}");
            }

            AssetDatabase.Refresh();
        }

        public static void DeleteEmptyFolders(List<DefaultAsset> folders)
        {
            var checkedFolders = new HashSet<string>();

            foreach (var folder in folders)
            {
                if (folder == null) continue;
                string rootPath = AssetDatabase.GetAssetPath(folder);
                DeleteEmptyFoldersRecursive(rootPath, checkedFolders);
            }

            AssetDatabase.Refresh();
        }

        private static void DeleteEmptyFoldersRecursive(string folderPath, HashSet<string> checkedFolders)
        {
            if (checkedFolders.Contains(folderPath)) return;
            checkedFolders.Add(folderPath);

            var subFolders = Directory.GetDirectories(folderPath);
            foreach (var sub in subFolders)
            {
                DeleteEmptyFoldersRecursive(sub.Replace("\\", "/"), checkedFolders);
            }

            var files = Directory.GetFiles(folderPath).Where(f => !f.EndsWith(".meta")).ToArray();
            var subDirs = Directory.GetDirectories(folderPath);

            if (files.Length == 0 && subDirs.Length == 0)
            {
                if (AssetDatabase.DeleteAsset(folderPath))
                    Debug.Log($"Deleted empty folder: {folderPath}");
            }
        }

        public static Dictionary<string, long> GetAssetSizes(IEnumerable<string> assetPaths)
        {
            Dictionary<string, long> sizes = new();

            foreach (var path in assetPaths)
            {
                try
                {
                    if (File.Exists(path))
                    {
                        sizes[path] = new FileInfo(path).Length;
                    }
                    else
                    {
                        sizes[path] = 0;
                    }
                }
                catch
                {
                    sizes[path] = 0;
                }
            }

            return sizes;
        }

    }
}