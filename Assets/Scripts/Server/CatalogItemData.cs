using System;

[Serializable]
public class CatalogItemData
{
    public string Id;
    public string Name;
    public string Category;
    public int Price;

    /// <summary>
    /// 선택적으로 아이템 프리팹, 주소 가능. Firestore에 없다면 비워두세요.
    /// </summary>
    public string PrefabAddress;

    /// <summary>
    /// 필요하다면 카탈로그에서 추가 정보를 보여줄 때 사용합니다.
    /// </summary>
    public string Description;

    public bool IsFree => Price <= 0;
}
