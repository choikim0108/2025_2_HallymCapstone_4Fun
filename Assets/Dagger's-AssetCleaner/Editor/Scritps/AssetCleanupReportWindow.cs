using UnityEditor;
using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Dagger.AssetCleanupTool 
{ 
    public class AssetCleanupReportWindow : EditorWindow
    {
        private Dictionary<string, List<string>> groupedAssets = new();
        private Dictionary<string, long> assetSizes = new();
        private Dictionary<string, bool> groupFoldouts = new();
        private string sortColumn = "Filename";
        private bool sortAscending = true;
        private Vector2 scroll;

        public static void ShowReport(IEnumerable<string> assetPaths)
        {
            var window = GetWindow<AssetCleanupReportWindow>("Simulated Deletion Report");
            window.BuildReport(assetPaths.ToList());
            window.Show();
        }

        private void BuildReport(List<string> assets)
        {
            groupedAssets = GroupByType(assets);
            assetSizes = AssetCleanup.GetAssetSizes(assets);
            groupFoldouts.Clear();

            foreach (var group in groupedAssets)
                groupFoldouts[group.Key] = true;
        }

        private void OnGUI()
        {
            if (groupedAssets == null || groupedAssets.Count == 0)
            {
                EditorGUILayout.LabelField("No assets to report.");
                return;
            }

            long totalSize = assetSizes.Values.Sum();
            int totalCount = assetSizes.Count;

            EditorGUILayout.LabelField($"Simulated Deletion Report", EditorStyles.boldLabel);
            EditorGUILayout.LabelField($"Total Assets: {totalCount}");
            EditorGUILayout.LabelField($"Estimated Disk Space: {FormatBytes(totalSize)}");

            scroll = EditorGUILayout.BeginScrollView(scroll);
            foreach (var group in groupedAssets)
            {
                groupFoldouts.TryGetValue(group.Key, out bool foldout);
                foldout = EditorGUILayout.Foldout(foldout, $"{group.Key} ({group.Value.Count})", true);
                groupFoldouts[group.Key] = foldout;

                if (!foldout) continue;

                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button("Filename", EditorStyles.boldLabel, GUILayout.Width(200)))
                    ToggleSort("Filename");
                if (GUILayout.Button("Path", EditorStyles.boldLabel, GUILayout.Width(300)))
                    ToggleSort("Path");
                if (GUILayout.Button("Size", EditorStyles.boldLabel, GUILayout.Width(80)))
                    ToggleSort("Size");
                EditorGUILayout.EndHorizontal();

                var sorted = Sort(group.Value);

                foreach (var path in sorted)
                {
                    string filename = Path.GetFileName(path);
                    string sizeStr = assetSizes.TryGetValue(path, out long size) ? FormatBytes(size) : "N/A";

                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.LabelField(filename, GUILayout.Width(200));
                    EditorGUILayout.LabelField(path, GUILayout.Width(300));
                    EditorGUILayout.LabelField(sizeStr, GUILayout.Width(80));
                    EditorGUILayout.EndHorizontal();
                }

                EditorGUILayout.Space(5);
            }
            EditorGUILayout.EndScrollView();
        }

        private List<string> Sort(List<string> input)
        {
            switch (sortColumn)
            {
                case "Filename":
                    return sortAscending
                        ? input.OrderBy(p => Path.GetFileName(p)).ToList()
                        : input.OrderByDescending(p => Path.GetFileName(p)).ToList();
                case "Path":
                    return sortAscending
                        ? input.OrderBy(p => p).ToList()
                        : input.OrderByDescending(p => p).ToList();
                case "Size":
                    return sortAscending
                        ? input.OrderBy(p => assetSizes.TryGetValue(p, out long s) ? s : 0).ToList()
                        : input.OrderByDescending(p => assetSizes.TryGetValue(p, out long s) ? s : 0).ToList();
            }
            return input;
        }

        private void ToggleSort(string column)
        {
            if (sortColumn == column)
                sortAscending = !sortAscending;
            else
            {
                sortColumn = column;
                sortAscending = true;
            }
        }

        private Dictionary<string, List<string>> GroupByType(IEnumerable<string> paths)
        {
            var result = new Dictionary<string, List<string>>();

            foreach (var path in paths)
            {
                string type = GetTypeForExtension(Path.GetExtension(path).ToLower());
                if (!result.ContainsKey(type))
                    result[type] = new List<string>();
                result[type].Add(path);
            }

            return result;
        }

        private string GetTypeForExtension(string ext)
        {
            return ext switch
            {
                ".png" or ".jpg" or ".jpeg" or ".tga" => "Textures",
                ".fbx" or ".obj" => "Models",
                ".mp3" or ".wav" or ".ogg" => "Audio",
                ".mat" => "Materials",
                ".shader" or ".compute" => "Shaders",
                ".prefab" => "Prefabs",
                _ => "Others"
            };
        }

        private string FormatBytes(long bytes)
        {
            if (bytes > 1_000_000)
                return (bytes / 1_000_000f).ToString("F2") + " MB";
            if (bytes > 1_000)
                return (bytes / 1_000f).ToString("F1") + " KB";
            return bytes + " B";
        }
    }
}