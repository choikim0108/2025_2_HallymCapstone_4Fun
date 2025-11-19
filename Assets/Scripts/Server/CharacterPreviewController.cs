using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

[DisallowMultipleComponent]
public class CharacterPreviewController : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField] private CustomizeManager customizeManager;

    [Header("Options")]
    [SerializeField] private bool applyOnEnable = true;

    private readonly Dictionary<string, string> currentLoadout = new Dictionary<string, string>();
    private bool applyInProgress;
    private bool applyQueued;

    private void OnEnable()
    {
        if (applyOnEnable && currentLoadout.Count > 0)
        {
            QueueApply();
        }
    }

    public void ApplyLoadout(Dictionary<string, string> loadout, bool replaceAll)
    {
        if (replaceAll)
        {
            currentLoadout.Clear();
        }

        if (loadout != null)
        {
            foreach (KeyValuePair<string, string> entry in loadout)
            {
                if (string.IsNullOrWhiteSpace(entry.Key))
                {
                    continue;
                }

                if (string.IsNullOrWhiteSpace(entry.Value))
                {
                    currentLoadout.Remove(entry.Key);
                }
                else
                {
                    currentLoadout[entry.Key] = entry.Value;
                }
            }
        }

        QueueApply();
    }

    public void ApplyPreviewItem(string category, string itemId)
    {
        if (string.IsNullOrWhiteSpace(category))
        {
            return;
        }

        if (string.IsNullOrWhiteSpace(itemId))
        {
            currentLoadout.Remove(category);
        }
        else
        {
            currentLoadout[category] = itemId;
        }

        QueueApply();
    }

    public Dictionary<string, string> GetCurrentPreviewLoadout()
    {
        return currentLoadout.ToDictionary(pair => pair.Key, pair => pair.Value);
    }

    private void QueueApply()
    {
        if (customizeManager == null)
        {
            Debug.LogWarning("CharacterPreviewController: CustomizeManager가 지정되지 않았습니다.");
            return;
        }

        if (applyInProgress)
        {
            applyQueued = true;
            return;
        }

        _ = ApplyToCharacterAsync();
    }

    private async Task ApplyToCharacterAsync()
    {
        applyInProgress = true;

        try
        {
            Dictionary<string, object> payload = currentLoadout.ToDictionary(pair => pair.Key, pair => (object)pair.Value);
            await customizeManager.ApplyLoadoutAsync(payload);
        }
        catch (Exception ex)
        {
            Debug.LogError($"CharacterPreviewController: 커스터마이징 적용 중 오류 발생 - {ex.Message}");
        }
        finally
        {
            applyInProgress = false;

            if (applyQueued)
            {
                applyQueued = false;
                QueueApply();
            }
        }
    }
}
