import requests
import pandas as pd
import json


url = "https://shop.arianagrande.com/products.json?limit=250"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

res = requests.get(url, headers=headers)

if res.status_code == 200:
    products = res.json().get("products", [])
    
    data = []
    for item in products:
        title = item.get("title")
        handle = item.get("handle")
        product_type = item.get("product_type")
        product_url = f"https://shop.arianagrande.com/products/{handle}"
        
        # Lấy giá trị của biến thể đầu tiên (variant)
        variants = item.get("variants", [])
        price = variants[0].get("price") if variants else "N/A"
        sku = variants[0].get("sku") if variants else "N/A"
        
        data.append({
            "title": title,
            "price": price,
            "type": product_type,
            "sku": sku,
            "url": product_url
        })
        
    #CSV
    df = pd.DataFrame(data)
    df.to_csv("data.csv", index=False, encoding="utf-8-sig")
    
    #TSV
    df.to_csv("data.tsv", sep="\t", index=False, encoding="utf-8-sig")
    
    #JSON
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"Crawl thành công {len(data)} sản phẩm.")
else:
    print("Lỗi kết nối:", res.status_code)