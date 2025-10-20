using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class DialogueOption
{
    public string optionId; // optional id
    public string text;
    public string nextNodeId; // empty = end
    public OptionAction action = OptionAction.None;
    public string actionParam; // parameter for action (eg. target name)
}

[Serializable]
public class DialogueNode
{
    public string nodeId;
    [TextArea(2,6)]
    public string text;
    public List<DialogueOption> options = new List<DialogueOption>();
}

public enum OptionAction
{
    None,
    CheckEquippedTargetEquals, // checks PlayerInteraction.equippedTarget contains actionParam (normalized, case-insensitive)
    SetEquippedTarget,         // sets PlayerInteraction.equippedTarget = actionParam
    ClearEquippedTarget,       // clears equipped target
    TriggerMissionSuccess,     // invokes MissionManager.MissionSuccess(actionParam)
}

[CreateAssetMenu(fileName = "NPCDialogueData", menuName = "Interaction/NPC Dialogue Data")]
public class NPCDialogueData : ScriptableObject
{
    public string npcName; // should match NPC GameObject name or a key
    public List<DialogueNode> nodes = new List<DialogueNode>();

    // helper lookup
    public DialogueNode GetNode(string id)
    {
        if (string.IsNullOrEmpty(id)) return null;
        return nodes.Find(n => n.nodeId == id);
    }

    public DialogueNode GetStartNode()
    {
        if (nodes == null || nodes.Count == 0) return null;
        return nodes[0];
    }
}
