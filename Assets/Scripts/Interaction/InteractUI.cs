
using UnityEngine;
using TMPro;
using UnityEngine.UI;

namespace Interaction
{

    public class InteractUI : MonoBehaviour
    {
    [Header("UI text")] public TextMeshProUGUI messageText;
    [Header("UI Panel")] public RectTransform panelRect;

    private Transform targetTransform;
    private Canvas parentCanvas;

        // 메시지 설정: 예시 - "Press F to interact\n\"오브젝트이름\""
        // ※ 텍스트 오브젝트에 Content Size Fitter(Vertical Fit: Preferred) 추가 권장
    public void SetMessage(string objectName)
        {
            if (messageText != null)
            {
                messageText.text = $"Press F to interact\n\"{objectName}\"";
                // 텍스트 길이에 맞춰 Panel 크기 자동 조정
                if (panelRect != null)
                {
                    LayoutRebuilder.ForceRebuildLayoutImmediate(messageText.rectTransform);
                    Vector2 size = messageText.rectTransform.sizeDelta;
                    // 패딩값(여유) 추가
                    float paddingX = 40f;
                    float paddingY = 20f;
                    panelRect.sizeDelta = new Vector2(size.x + paddingX, size.y + paddingY);
                }
            }
        }

        // 타겟 Transform과 Canvas를 저장
        public void SetTarget(Transform target, Canvas canvas)
        {
            targetTransform = target;
            parentCanvas = canvas;
        }

        private Vector2 lastAnchoredPosition;
        public float followSmooth = 10f; // UI 따라가기 부드러움 계수

        void LateUpdate()
        {
            if (targetTransform != null && parentCanvas != null)
            {
                SetWorldPosition(targetTransform.position + Vector3.up * 1.5f, parentCanvas);
            }
        }

        // UI를 월드 오브젝트 위에 위치시키는 함수
        public void SetWorldPosition(Vector3 worldPos, Canvas canvas)
        {
            if (canvas == null || panelRect == null) return;
            Camera cam = Camera.main;
            if (cam == null) return;
            Vector2 screenPoint = RectTransformUtility.WorldToScreenPoint(cam, worldPos);
            Vector2 localPoint;
            if (RectTransformUtility.ScreenPointToLocalPointInRectangle(canvas.transform as RectTransform, screenPoint, null, out localPoint))
            {
                if (lastAnchoredPosition == Vector2.zero)
                    lastAnchoredPosition = localPoint;
                lastAnchoredPosition = Vector2.Lerp(lastAnchoredPosition, localPoint, Time.deltaTime * followSmooth);
                panelRect.anchoredPosition = lastAnchoredPosition;
            }
        }
    }
}