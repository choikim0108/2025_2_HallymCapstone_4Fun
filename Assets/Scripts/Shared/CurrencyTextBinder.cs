using TMPro;
using UnityEngine;

/// <summary>
/// 씬별 통화 표시 TextMeshProUGUI를 CurrencyManager에 연결합니다.
/// </summary>
[RequireComponent(typeof(TextMeshProUGUI))]
public class CurrencyTextBinder : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI targetText;

    private void Awake()
    {
        if (targetText == null)
        {
            targetText = GetComponent<TextMeshProUGUI>();
        }
    }

    private void OnEnable()
    {
        CurrencyManager manager = CurrencyManager.Instance;
        if (manager != null)
        {
            manager.RegisterCurrencyText(targetText);
        }
    }

    private void OnDisable()
    {
        CurrencyManager manager = CurrencyManager.Instance;
        if (manager != null)
        {
            manager.UnregisterCurrencyText(targetText);
        }
    }
}
