using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class NPCDialogueSys : MonoBehaviour
{
    public static NPCDialogueSys Instance { get; private set; }

    [Header("UI References")]
    public GameObject dialogueUI; // assign in inspector or auto-find
    public TextMeshProUGUI npcNameText;
    public TextMeshProUGUI dialogueText;
    public Transform optionsRoot;
    public Button optionButtonPrefab;
    public Image npcPhotoImage;
    public Button exitButton;

    private NPCDialogueData currentData;
    private DialogueNode currentNode;
    private GameObject currentPlayerObj; // who opened dialogue

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        // try auto-find
        if (dialogueUI == null)
        {
            var canvas = GameObject.Find("Canvas");
            if (canvas != null)
            {
                var go = canvas.transform.Find("NPCDialogueUI");
                if (go != null)
                    dialogueUI = go.gameObject;
            }
        }
        if (dialogueUI != null)
        {
            if (npcPhotoImage == null)
            {
                var npcPhotoTransform = dialogueUI.transform.Find("NPCPhoto");
                if (npcPhotoTransform != null)
                    npcPhotoImage = npcPhotoTransform.GetComponent<Image>();
            }

            if (exitButton == null)
            {
                var exitButtonTransform = dialogueUI.transform.Find("ExitButton");
                if (exitButtonTransform != null)
                    exitButton = exitButtonTransform.GetComponent<Button>();
            }

            if (npcPhotoImage != null)
            {
                npcPhotoImage.sprite = null;
                npcPhotoImage.enabled = false;
            }

            dialogueUI.SetActive(false);
        }

        if (exitButton != null)
            exitButton.onClick.AddListener(HandleExitButtonClicked);
    }

    // Show dialogue for an NPC by name. playerObj is used to access PlayerInteraction.
    public void ShowDialogue(string npcName, GameObject playerObj, string displayNameOverride = null)
    {
        currentPlayerObj = playerObj;
            // mark player in dialogue so PlayerInteraction won't show interact UI while dialog is open
            var pip = playerObj != null ? playerObj.GetComponent<PlayerInteraction>() : null;
            if (pip != null)
            {
                pip.isInDialogue = true;
                // hide current target UI if any
                if (pip.currentTarget != null)
                    pip.currentTarget.InteractUIHide();
            }
        // load DialogueData from Resources/NPCDialogues/<npcName> or from Resources by name
        currentData = Resources.Load<NPCDialogueData>($"NPCDialogues/{npcName}");
        if (currentData == null)
        {
            currentData = Resources.Load<NPCDialogueData>(npcName);
        }

        if (currentData == null)
        {
            Debug.LogWarning($"[NPCDialogueSys] Dialogue data not found for '{npcName}'");
            return;
        }

        string resolvedDisplayName = !string.IsNullOrWhiteSpace(displayNameOverride)
            ? displayNameOverride
            : (!string.IsNullOrWhiteSpace(currentData.npcName) ? currentData.npcName : npcName);
        if (npcNameText != null)
            npcNameText.text = resolvedDisplayName;

        var start = currentData.GetStartNode();
        if (start == null)
        {
            Debug.LogWarning($"[NPCDialogueSys] No nodes in dialogue data for '{npcName}'");
            return;
        }

        dialogueUI?.SetActive(true);
        UpdateNpcPhoto(currentData.npcName, npcName);
        ShowNode(start);
    }

    public void CloseDialogue()
    {
        dialogueUI?.SetActive(false);
        ClearOptions();
        currentData = null;
        currentNode = null;
        if (npcPhotoImage != null)
        {
            npcPhotoImage.sprite = null;
            npcPhotoImage.enabled = false;
        }
        // clear player's dialogue flag if set
        if (currentPlayerObj != null)
        {
            var pip = currentPlayerObj.GetComponent<PlayerInteraction>();
            if (pip != null)
                pip.isInDialogue = false;
        }
        currentPlayerObj = null;
    }

    private void UpdateNpcPhoto(string preferredName, string fallbackName)
    {
        if (npcPhotoImage == null)
            return;

        Sprite sprite = LoadNpcPhotoSprite(preferredName);
        if (sprite == null)
            sprite = LoadNpcPhotoSprite(fallbackName);

        if (sprite != null)
        {
            npcPhotoImage.sprite = sprite;
            npcPhotoImage.enabled = true;
        }
        else
        {
            npcPhotoImage.sprite = null;
            npcPhotoImage.enabled = false;
            if (!string.IsNullOrWhiteSpace(preferredName) || !string.IsNullOrWhiteSpace(fallbackName))
            {
                Debug.LogWarning($"[NPCDialogueSys] NPC photo sprite not found in Resources/Images for '{preferredName ?? string.Empty}' (fallback '{fallbackName ?? string.Empty}').");
            }
        }
    }

    private Sprite LoadNpcPhotoSprite(string npcName)
    {
        if (string.IsNullOrWhiteSpace(npcName))
            return null;

        string trimmed = npcName.Trim();
        var visited = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var order = new List<string>();

        void AddCandidate(string candidate)
        {
            if (string.IsNullOrWhiteSpace(candidate))
                return;
            if (visited.Add(candidate))
                order.Add(candidate);
        }

        AddCandidate(trimmed);
        AddCandidate(trimmed.Replace(" ", ""));
        AddCandidate(trimmed.Replace(" ", "_"));

        string alnum = System.Text.RegularExpressions.Regex.Replace(trimmed, "[^a-zA-Z0-9_ ]", "");
        if (!string.Equals(alnum, trimmed, StringComparison.Ordinal))
        {
            AddCandidate(alnum);
            AddCandidate(alnum.Replace(" ", ""));
            AddCandidate(alnum.Replace(" ", "_"));
        }

        string lower = trimmed.ToLowerInvariant();
        AddCandidate(lower);
        AddCandidate(lower.Replace(" ", ""));
        AddCandidate(lower.Replace(" ", "_"));

        foreach (var candidate in order)
        {
            var sprite = Resources.Load<Sprite>($"Images/{candidate}");
            if (sprite != null)
                return sprite;
        }

        return null;
    }

    private void ShowNode(DialogueNode node)
    {
        currentNode = node;
        if (dialogueText != null) dialogueText.text = node.text;
        ClearOptions();
        // create buttons
        foreach (var opt in node.options)
        {
            var btn = Instantiate(optionButtonPrefab, optionsRoot);
            var txt = btn.GetComponentInChildren<TextMeshProUGUI>();
            if (txt != null) txt.text = opt.text;
            btn.onClick.AddListener(() => OnOptionSelected(opt));
        }
    }

    private void ClearOptions()
    {
        if (optionsRoot == null) return;
        for (int i = optionsRoot.childCount - 1; i >= 0; i--)
        {
            Destroy(optionsRoot.GetChild(i).gameObject);
        }
    }

    private void OnOptionSelected(DialogueOption option)
    {
        Debug.Log($"[NPCDialogueSys] Option selected: text='{option.text}', action={option.action}, actionParam={option.actionParam}, nextNodeId={option.nextNodeId}");
        // perform action if any
        if (currentPlayerObj != null && option.action != OptionAction.None)
        {
            var pi = currentPlayerObj.GetComponent<PlayerInteraction>();
            if (pi != null)
            {
                switch (option.action)
                {
                    case OptionAction.CheckEquippedTargetEquals:
                        {
                            string playerVal = pi.equippedTarget ?? "";
                            string param = option.actionParam ?? "";

                            // remove common suffixes like " (1)", " (Clone)" and trim
                            string StripSuffixes(string s)
                            {
                                if (string.IsNullOrEmpty(s)) return "";
                                // remove (n) numeric suffixes
                                s = System.Text.RegularExpressions.Regex.Replace(s, "\\s*\\(\\d+\\)\\s*$", "");
                                // remove (Clone) case-insensitive
                                s = System.Text.RegularExpressions.Regex.Replace(s, "\\(clone\\)", "", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
                                return s.Trim();
                            }

                            // Tokenize: split camelCase, replace non-alphanumeric with space, lowercase tokens
                            string[] Tokenize(string s)
                            {
                                s = StripSuffixes(s);
                                if (string.IsNullOrEmpty(s)) return new string[0];
                                // insert space between lower->upper (camel case) e.g. ExplosiveBox -> Explosive Box
                                s = System.Text.RegularExpressions.Regex.Replace(s, "([a-z])([A-Z])", "$1 $2");
                                // replace non-alphanumeric with space
                                s = System.Text.RegularExpressions.Regex.Replace(s, "[^a-zA-Z0-9]+", " ");
                                var parts = s.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                                for (int i = 0; i < parts.Length; i++) parts[i] = parts[i].ToLowerInvariant();
                                return parts;
                            }

                            var playerTokens = Tokenize(playerVal);
                            var paramTokens = Tokenize(param);

                            bool match = false;
                            if (paramTokens.Length == 0)
                            {
                                match = false;
                            }
                            else
                            {
                                // check that every param token exists in player tokens
                                match = true;
                                foreach (var t in paramTokens)
                                {
                                    bool found = false;
                                    foreach (var pt in playerTokens)
                                    {
                                        if (pt == t)
                                        {
                                            found = true; break;
                                        }
                                    }
                                    if (!found)
                                    {
                                        match = false; break;
                                    }
                                }
                                // fallback: if exact token subset failed, try normalized contains check
                                if (!match)
                                {
                                    var nPlayer = string.Join("", playerTokens);
                                    var nParam = string.Join("", paramTokens);
                                    if (!string.IsNullOrEmpty(nParam) && !string.IsNullOrEmpty(nPlayer))
                                        match = nPlayer.Contains(nParam) || nParam.Contains(nPlayer);
                                }
                            }

                            Debug.Log($"[NPCDialogueSys] CheckEquippedTargetEquals(tokens): playerTokens=[{string.Join(",", playerTokens)}], paramTokens=[{string.Join(",", paramTokens)}], match={match}");
                            if (!match)
                            {
                                CloseDialogue();
                                return;
                            }
                        }
                        break;
                    case OptionAction.SetEquippedTarget:
                        pi.equippedTarget = option.actionParam;
                        Debug.Log($"[NPCDialogueSys] SetEquippedTarget: set to '{option.actionParam}'");
                        break;
                    case OptionAction.ClearEquippedTarget:
                        pi.equippedTarget = null;
                        Debug.Log("[NPCDialogueSys] ClearEquippedTarget: cleared");
                        break;
                    case OptionAction.TriggerMissionSuccess:
                        {
                            var manager = MissionManager.Instance ?? GameObject.FindFirstObjectByType<MissionManager>();
                            if (manager != null)
                            {
                                manager.MissionSuccess(string.IsNullOrWhiteSpace(option.actionParam) ? null : option.actionParam);
                            }
                            else
                            {
                                Debug.LogWarning("[NPCDialogueSys] MissionManager not found for TriggerMissionSuccess action");
                            }
                        }
                        break;
                }
            }
        }

        // navigate to next node or close
        if (!string.IsNullOrEmpty(option.nextNodeId) && currentData != null)
        {
            var next = currentData.GetNode(option.nextNodeId);
            if (next != null)
            {
                ShowNode(next);
                return;
            }
        }

        CloseDialogue();
    }

    private void HandleExitButtonClicked()
    {
        CloseDialogue();
    }

    private void OnDestroy()
    {
        if (exitButton != null)
            exitButton.onClick.RemoveListener(HandleExitButtonClicked);
    }
}
