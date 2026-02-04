"""
見開きページスキャナー
本の見開き画像から左右のページを個別に切り出してPDFを作成するアプリケーション
4点のホモグラフィ変換で透視補正
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from typing import List, Tuple, Optional


class PagePointSelector:
    """画像上で4点を選択するためのクラス"""
    
    def __init__(self, image: np.ndarray, window_name: str):
        self.image = image.copy()
        self.display_image = image.copy()
        self.window_name = window_name
        self.points = []
        self.max_points = 4
        
    def mouse_callback(self, event, x, y, flags, param):
        """マウスクリックイベントのコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < self.max_points:
            # 点を追加
            self.points.append((x, y))
            
            # 画像を更新して点を描画
            self.display_image = self.image.copy()
            for i, point in enumerate(self.points):
                cv2.circle(self.display_image, point, 5, (0, 255, 0), -1)
                cv2.putText(self.display_image, str(i+1), 
                           (point[0] + 10, point[1] + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 4点揃ったら線を引く
            if len(self.points) == 4:
                pts = np.array(self.points, dtype=np.int32)
                cv2.polylines(self.display_image, [pts], True, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, self.display_image)
    
    def select_points(self) -> List[Tuple[int, int]]:
        """ユーザーに4点を選択させる"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, self.display_image)
        
        print(f"\n{self.window_name}: 4つの角をクリックしてください（左上→右上→右下→左下の順）")
        print("完了したらEnterキーを押してください。やり直す場合はRキーを押してください。")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Enterキーで確定（4点選択済みの場合）
            if key == 13 and len(self.points) == 4:
                break
            
            # Rキーでリセット
            if key == ord('r'):
                self.points = []
                self.display_image = self.image.copy()
                cv2.imshow(self.window_name, self.display_image)
                print("リセットしました。もう一度4点を選択してください。")
        
        cv2.destroyWindow(self.window_name)
        return self.points


def perspective_transform_page(image: np.ndarray, src_points: List[Tuple[int, int]], 
                               output_width: int = 800, output_height: int = 1200) -> np.ndarray:
    """
    4点のホモグラフィ変換で、任意の四角形を長方形に変換
    
    Args:
        image: 入力画像
        src_points: 元画像の4点（左上、右上、右下、左下の順）
        output_width: 出力画像の幅
        output_height: 出力画像の高さ
    
    Returns:
        変換後の画像
    """
    if len(src_points) != 4:
        raise ValueError("4点を指定してください")
    
    # 元画像の4点
    src = np.float32(src_points)
    
    # 出力画像の4点（長方形）
    dst = np.float32([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ])
    
    # ホモグラフィ行列を計算
    matrix = cv2.getPerspectiveTransform(src, dst)
    
    # 透視変換を適用
    warped = cv2.warpPerspective(image, matrix, (output_width, output_height))
    
    return warped


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    影除去と二値化で画像を改善
    
    照明のムラを除去して文字を読みやすくします。
    
    Args:
        image: 入力画像（BGR）
    
    Returns:
        影が除去され読みやすくなった画像
    """
    # 1. 影（照明成分）を推定する
    # 強力なガウスぼかしをかけることで、文字を消して「明るさのムラ」だけを抽出します
    dilated_img = cv2.dilate(image, np.ones((7, 7), np.uint8))  # 文字を少し太らせて消えやすくする
    bg_img = cv2.medianBlur(dilated_img, 21)  # 中央値フィルタでノイズ除去
    bg_img = cv2.GaussianBlur(bg_img, (51, 51), 0)  # 大きくぼかして背景を作る
    
    # 2. 元画像から背景成分を除去する（除算）
    # 「元画像 / 背景画像」という計算をすることで、ムラがキャンセルされます
    result = cv2.divide(image, bg_img, scale=255)
    
    # 3. 少しだけコントラストを強調して文字を濃くする
    # これにより、紙の質感を生かしつつ、読みやすくします
    final = cv2.convertScaleAbs(result, alpha=1.1, beta=-20)
    
    return final


def crop_bounding_rect(image: np.ndarray, src_points: List[Tuple[int, int]]) -> np.ndarray:
    """
    指定された4点を包含する矩形領域を切り出す（透視変換なし）
    
    Args:
        image: 入力画像
        src_points: 切り出し領域の4点（順序は問わない）
    
    Returns:
        切り出された画像
    """
    # 4点からバウンディングボックスを計算
    points = np.array(src_points)
    x_min = int(np.min(points[:, 0]))
    x_max = int(np.max(points[:, 0]))
    y_min = int(np.min(points[:, 1]))
    y_max = int(np.max(points[:, 1]))
    
    # 画像の範囲内に収める
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)
    
    # 矩形領域を切り出し
    cropped = image[y_min:y_max, x_min:x_max]
    
    return cropped


def interpolate_points(points_start: List[Tuple[int, int]], 
                       points_end: List[Tuple[int, int]], 
                       t: float) -> List[Tuple[int, int]]:
    """
    2つの4点セットを線形補間する
    
    Args:
        points_start: 最初の4点
        points_end: 最終の4点
        t: 補間係数（0.0 = 最初、1.0 = 最終）
    
    Returns:
        補間された4点
    """
    interpolated = []
    for i in range(4):
        x = int(points_start[i][0] * (1 - t) + points_end[i][0] * t)
        y = int(points_start[i][1] * (1 - t) + points_end[i][1] * t)
        interpolated.append((x, y))
    return interpolated


def get_page_regions(image_path: Path, label: str = "ページ") -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    画像から左右のページ領域（4点）を取得
    
    Args:
        image_path: 画像ファイルのパス
        label: 画面に表示するラベル（例："最初のページ"、"最終ページ"）
    
    Returns:
        (左ページ4点, 右ページ4点)のタプル
    """
    # 画像を読み込み
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"画像を読み込めません: {image_path}")
    
    # 表示用に画像をリサイズ（画面に収まるように）
    display_height = 800
    aspect_ratio = image.shape[1] / image.shape[0]
    display_width = int(display_height * aspect_ratio)
    display_image = cv2.resize(image, (display_width, display_height))
    
    # 元の画像サイズとの比率を計算
    scale_x = image.shape[1] / display_width
    scale_y = image.shape[0] / display_height
    
    # 左ページの4点を選択
    left_selector = PagePointSelector(display_image, f"{label} - 左ページの選択")
    left_points_display = left_selector.select_points()
    
    # 右ページの4点を選択
    right_selector = PagePointSelector(display_image, f"{label} - 右ページの選択")
    right_points_display = right_selector.select_points()
    
    # 選択した点を元の画像サイズにスケール
    left_points = [(int(x * scale_x), int(y * scale_y)) for x, y in left_points_display]
    right_points = [(int(x * scale_x), int(y * scale_y)) for x, y in right_points_display]
    
    return left_points, right_points


def process_spread_image_with_regions(image_path: Path, 
                                       left_points: List[Tuple[int, int]], 
                                       right_points: List[Tuple[int, int]],
                                       page_width: int = 800,
                                       page_height: int = 1200) -> Tuple[np.ndarray, np.ndarray]:
    """
    指定された4点を使ってホモグラフィ変換で見開き画像から左右のページを切り出す
    コントラスト改善を適用します
    
    Args:
        image_path: 画像ファイルのパス
        left_points: 左ページの4点
        right_points: 右ページの4点
        page_width: ページの幅
        page_height: ページの高さ
    
    Returns:
        (左ページ画像, 右ページ画像)のタプル
    """
    # 画像を読み込み
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"画像を読み込めません: {image_path}")
    
    # ホモグラフィ変換で透視補正
    left_page = perspective_transform_page(image, left_points, page_width, page_height)
    right_page = perspective_transform_page(image, right_points, page_width, page_height)
    
    # コントラストを改善
    left_page = enhance_contrast(left_page)
    right_page = enhance_contrast(right_page)
    
    return left_page, right_page


def process_all_images(input_folder: Path, output_pdf: Path, 
                       page_width: int = 800, page_height: int = 1200):
    """
    フォルダ内の全画像を処理してPDFを作成
    最初の画像と最終ページで領域を指定し、中間ページは線形補間で領域を自動調整
    
    Args:
        input_folder: 入力画像フォルダ
        output_pdf: 出力PDFファイルのパス
        page_width: ページの幅（未使用、互換性のため残す）
        page_height: ページの高さ（未使用、互換性のため残す）
    """
    # 対応する画像拡張子
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 画像ファイルを取得してソート
    image_files = sorted([
        f for f in input_folder.iterdir() 
        if f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        print(f"エラー: {input_folder} に画像ファイルが見つかりません")
        return
    
    print(f"\n処理対象: {len(image_files)}個の画像ファイル")
    
    # 最初の画像で左右のページ領域を取得
    print(f"\n【最初のページ】{image_files[0].name}")
    print("最初の画像で領域を指定してください")
    try:
        left_points_start, right_points_start = get_page_regions(image_files[0], "最初のページ")
        print(f"✓ 最初のページの領域を取得しました")
        print(f"  左ページ: {left_points_start}")
        print(f"  右ページ: {right_points_start}")
    except Exception as e:
        print(f"✗ エラー: 領域の取得に失敗しました - {e}")
        return
    
    # 最終ページの領域を取得
    print(f"\n【最終ページ】{image_files[-1].name}")
    print("最終ページで領域を指定してください")
    try:
        left_points_end, right_points_end = get_page_regions(image_files[-1], "最終ページ")
        print(f"✓ 最終ページの領域を取得しました")
        print(f"  左ページ: {left_points_end}")
        print(f"  右ページ: {right_points_end}")
    except Exception as e:
        print(f"✗ エラー: 領域の取得に失敗しました - {e}")
        return
    
    # 全ページを格納するリスト
    all_pages = []
    
    # 各画像を処理
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {image_file.name} を処理中...")
        
        try:
            # 補間係数を計算（0.0 = 最初、1.0 = 最終）
            if len(image_files) == 1:
                t = 0.0
            else:
                t = (i - 1) / (len(image_files) - 1)
            
            # 現在のページの領域を線形補間で計算
            left_points = interpolate_points(left_points_start, left_points_end, t)
            right_points = interpolate_points(right_points_start, right_points_end, t)
            
            # 補間された領域を使って左右のページを切り出し
            left_page, right_page = process_spread_image_with_regions(
                image_file, left_points, right_points, 
                page_width, page_height
            )
            
            # BGRからRGBに変換（PillowはRGBを使用）
            left_page_rgb = cv2.cvtColor(left_page, cv2.COLOR_BGR2RGB)
            right_page_rgb = cv2.cvtColor(right_page, cv2.COLOR_BGR2RGB)
            
            # PIL Imageに変換
            left_pil = Image.fromarray(left_page_rgb)
            right_pil = Image.fromarray(right_page_rgb)
            
            # リストに追加（左ページ→右ページの順）
            all_pages.extend([left_pil, right_pil])
            
            print(f"✓ 完了: {len(all_pages)}ページ追加 (t={t:.3f})")
            
        except Exception as e:
            print(f"✗ エラー: {image_file.name} の処理に失敗しました - {e}")
            continue
    
    # PDFとして保存
    if all_pages:
        print(f"\nPDFを作成中: {output_pdf}")
        all_pages[0].save(
            output_pdf,
            save_all=True,
            append_images=all_pages[1:],
            resolution=100.0,
            quality=95,
            optimize=False
        )
        print(f"✓ 完了! {len(all_pages)}ページのPDFを作成しました: {output_pdf}")
    else:
        print("エラー: 処理できた画像がありません")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="見開きページスキャナー - 本の見開き画像から左右のページを切り出してPDFを作成"
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="見開き画像が含まれるフォルダのパス"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.pdf",
        help="出力PDFファイル名（デフォルト: output.pdf）"
    )
    parser.add_argument(
        "-w", "--width",
        type=int,
        default=800,
        help="（非推奨：互換性のため残されています）"
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=1200,
        help="（非推奨：互換性のため残されています）"
    )
    
    args = parser.parse_args()
    
    # パスを作成
    input_folder = Path(args.input_folder)
    output_pdf = Path(args.output)
    
    # フォルダの存在確認
    if not input_folder.exists():
        print(f"エラー: フォルダが見つかりません: {input_folder}")
        return
    
    if not input_folder.is_dir():
        print(f"エラー: パスがフォルダではありません: {input_folder}")
        return
    
    # 処理を実行
    process_all_images(input_folder, output_pdf, args.width, args.height)


if __name__ == "__main__":
    main()
