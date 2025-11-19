using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace Dagger.AssetCleanupTool 
{ 
    public class AssetCleanupToolEditor : EditorWindow
    {
        private List<DefaultAsset> selectedFolders = new();
        private List<SceneAsset> selectedScenes = new();
        private List<string> unusedAssets = new();
        private Dictionary<string, long> assetSizes = new();
        private HashSet<string> selectedForDeletion = new();

        private List<string> cachedFilteredAssets = new();
        private long cachedSelectedSize = 0;
        private bool needsRebuildFilteredAssets = true;

        private Vector2 scroll;
        private bool selectAllToggle = false;
        private bool simulateDeletion = true;

        private string searchQuery = "";
        private bool filtersFoldout = true;
        private bool filterTextures = true;
        private bool filterModels = true;
        private bool filterAudio = true;
        private bool filterMaterials = true;
        private bool filterShaders = true;
        private bool filterPrefabs = true;
        private bool filterOthers = true;

        [MenuItem("Tools/Asset Cleanup Tool")]
        public static void ShowWindow()
        {
            GetWindow<AssetCleanupToolEditor>("Asset Cleanup Tool");
        }

        private void OnGUI()
        {
            GUILayout.Label("Unused Asset Cleanup", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            // Folders
            EditorGUILayout.LabelField("Folders to Scan:", EditorStyles.boldLabel);
            for (int i = 0; i < selectedFolders.Count; i++)
            {
                EditorGUILayout.BeginHorizontal();
                selectedFolders[i] = (DefaultAsset)EditorGUILayout.ObjectField(selectedFolders[i], typeof(DefaultAsset), false);
                if (GUILayout.Button("X", GUILayout.Width(20))) { selectedFolders.RemoveAt(i); i--; }
                EditorGUILayout.EndHorizontal();
            }
            if (GUILayout.Button("Add Folder")) selectedFolders.Add(null);

            EditorGUILayout.Space();

            // Scenes
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Scenes to Include in Reference Check:", EditorStyles.boldLabel);
            if (GUILayout.Button("Auto Add Build Scenes", GUILayout.Width(140))) AddBuildScenesToList();
            EditorGUILayout.EndHorizontal();

            for (int i = 0; i < selectedScenes.Count; i++)
            {
                EditorGUILayout.BeginHorizontal();
                selectedScenes[i] = (SceneAsset)EditorGUILayout.ObjectField(selectedScenes[i], typeof(SceneAsset), false);
                if (GUILayout.Button("X", GUILayout.Width(20))) { selectedScenes.RemoveAt(i); i--; }
                EditorGUILayout.EndHorizontal();
            }
            if (GUILayout.Button("Add Scene")) selectedScenes.Add(null);

            EditorGUILayout.Space();

            if (GUILayout.Button("Scan for Unused Assets"))
            {
                if (SceneManager.GetActiveScene().isDirty)
                {
                    bool save = EditorUtility.DisplayDialog("Unsaved Scene",
                        "You have unsaved changes. Save before scanning?", "Save and Continue", "Cancel");
                    if (!save) return;
                    EditorSceneManager.SaveScene(SceneManager.GetActiveScene());
                }

                unusedAssets = AssetCleanup.ScanUnusedAssets(selectedScenes, selectedFolders);
                assetSizes = AssetCleanup.GetAssetSizes(unusedAssets);
                selectedForDeletion.Clear();
                selectAllToggle = false;
                needsRebuildFilteredAssets = true;
            }

            EditorGUILayout.Space();
            GUILayout.Label("Search & Filters", EditorStyles.boldLabel);
            string newSearchQuery = EditorGUILayout.TextField("Search:", searchQuery);
            if (newSearchQuery != searchQuery)
            {
                searchQuery = newSearchQuery;
                needsRebuildFilteredAssets = true;
            }

            filtersFoldout = EditorGUILayout.Foldout(filtersFoldout, "Filters");
            if (filtersFoldout)
            {
                EditorGUI.indentLevel++;
                if (DrawFilterToggle(ref filterTextures, "Textures (.png, .jpg, .tga)")) needsRebuildFilteredAssets = true;
                if (DrawFilterToggle(ref filterModels, "Models (.fbx, .obj)")) needsRebuildFilteredAssets = true;
                if (DrawFilterToggle(ref filterAudio, "Audio (.mp3, .wav, .ogg)")) needsRebuildFilteredAssets = true;
                if (DrawFilterToggle(ref filterMaterials, "Materials (.mat)")) needsRebuildFilteredAssets = true;
                if (DrawFilterToggle(ref filterShaders, "Shaders (.shader, .compute)")) needsRebuildFilteredAssets = true;
                if (DrawFilterToggle(ref filterPrefabs, "Prefabs (.prefab)")) needsRebuildFilteredAssets = true;
                if (DrawFilterToggle(ref filterOthers, "Other Types")) needsRebuildFilteredAssets = true;
                EditorGUI.indentLevel--;
            }

            if (needsRebuildFilteredAssets)
            {
                cachedFilteredAssets = AssetCleanup.FilterAssets(unusedAssets, searchQuery,
                    filterTextures, filterModels, filterAudio,
                    filterMaterials, filterShaders, filterPrefabs, filterOthers);

                cachedSelectedSize = selectedForDeletion
                    .Where(path => assetSizes.ContainsKey(path))
                    .Sum(path => assetSizes[path]);

                needsRebuildFilteredAssets = false;
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Unused Assets Found: " + cachedFilteredAssets.Count, EditorStyles.boldLabel);

            // Summary side-by-side
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField($"Selected Assets: {selectedForDeletion.Count}", GUILayout.Width(200));
            EditorGUILayout.LabelField($"Total Selected Size: {FormatBytes(cachedSelectedSize)}");
            EditorGUILayout.EndHorizontal();

            if (cachedFilteredAssets.Count > 0)
            {
                bool newSelectAllToggle = EditorGUILayout.ToggleLeft("Select All", selectAllToggle);
                if (newSelectAllToggle != selectAllToggle)
                {
                    selectAllToggle = newSelectAllToggle;
                    if (selectAllToggle)
                        selectedForDeletion = new HashSet<string>(cachedFilteredAssets);
                    else
                        selectedForDeletion.Clear();
                    needsRebuildFilteredAssets = true;
                }
            }

            scroll = EditorGUILayout.BeginScrollView(scroll);
            foreach (var asset in cachedFilteredAssets)
            {
                EditorGUILayout.BeginHorizontal();

                bool selected = selectedForDeletion.Contains(asset);
                bool newSelected = EditorGUILayout.Toggle(selected, GUILayout.Width(20));
                if (newSelected != selected)
                {
                    if (newSelected) selectedForDeletion.Add(asset);
                    else selectedForDeletion.Remove(asset);
                    needsRebuildFilteredAssets = true;
                }

                string filename = Path.GetFileName(asset);
                string sizeStr = assetSizes.TryGetValue(asset, out long size) ? FormatBytes(size) : "N/A";

                EditorGUILayout.LabelField(filename, GUILayout.Width(position.width * 0.5f));
                EditorGUILayout.LabelField(sizeStr, GUILayout.Width(80));

                EditorGUILayout.EndHorizontal();
            }
            EditorGUILayout.EndScrollView();

            EditorGUILayout.Space();
            simulateDeletion = EditorGUILayout.ToggleLeft("Simulate Deletion (Dry Run Mode)", simulateDeletion);

            GUI.enabled = selectedForDeletion.Count > 0;
            string buttonLabel = simulateDeletion ? "Simulate Deletion" : "Delete Selected Assets";
            GUI.backgroundColor = simulateDeletion ? Color.yellow : Color.red;

            if (GUILayout.Button(buttonLabel))
            {
                if (simulateDeletion)
                {
                    AssetCleanupReportWindow.ShowReport(selectedForDeletion);
                }
                else
                {
                    if (EditorUtility.DisplayDialog("Delete Assets",
                        $"Are you sure you want to delete {selectedForDeletion.Count} assets?",
                        "Yes, Delete", "Cancel"))
                    {
                        AssetCleanup.DeleteAssets(selectedForDeletion);
                        AssetCleanup.DeleteEmptyFolders(selectedFolders);

                        unusedAssets = AssetCleanup.ScanUnusedAssets(selectedScenes, selectedFolders);
                        assetSizes = AssetCleanup.GetAssetSizes(unusedAssets);
                        selectedForDeletion.Clear();
                        selectAllToggle = false;
                        needsRebuildFilteredAssets = true;
                    }
                }
            }

            GUI.backgroundColor = Color.white;
            GUI.enabled = true;
        }

        private bool DrawFilterToggle(ref bool field, string label)
        {
            bool newValue = EditorGUILayout.ToggleLeft(label, field);
            if (newValue != field)
            {
                field = newValue;
                return true;
            }
            return false;
        }

        private void AddBuildScenesToList()
        {
            var buildScenes = EditorBuildSettings.scenes
                .Where(s => s.enabled)
                .Select(s => s.path)
                .ToList();

            foreach (var scenePath in buildScenes)
            {
                var sceneAsset = AssetDatabase.LoadAssetAtPath<SceneAsset>(scenePath);
                if (sceneAsset != null && !selectedScenes.Contains(sceneAsset))
                    selectedScenes.Add(sceneAsset);
            }

            Debug.Log($"Added {buildScenes.Count} build scene(s).");
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