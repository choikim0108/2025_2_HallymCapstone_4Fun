using UnityEngine;
using Interaction;

public class BusInteractable : Interactable
{
    public override void Interact(PlayerInteraction player)
    {
        // PlayerInteraction 스크립트의 버스 탑승 시도 메서드를 호출
        player.AttemptToBoardBus(this.gameObject);
        InteractUIHide();
    }
}
