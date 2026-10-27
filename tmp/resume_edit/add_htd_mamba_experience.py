from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


SOURCE = Path(r"D:\QQQ-muban\tmp\resume_edit\source_resume.docx")
OUTPUT = Path(r"D:\QQQ-muban\output\吴嘉谦的个人简历_HTD-Mamba项目版.docx")

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
W = f"{{{W_NS}}}"

TITLE = "基于 HTD-Mamba 的无人机高光谱目标检测"
BULLETS = [
    "面向无人机高光谱目标检测场景，完成 .hdr/.spe 原始数据解析、高光谱立方体重建与暗场校正；开展像素级目标标注并转换为 map 矩阵，与 data 合并生成模型规范输入的 .mat 数据集，标注结果用于测试阶段性能验证。",
    "在 AutoDL 云服务器 V100 环境完成 PyTorch/CUDA 及 selective_scan_cuda、causal_conv1d 依赖配置，搭建 HTD-Mamba 训练与测试流程；基于自监督对比学习完成模型训练、最优权重筛选及整幅高光谱图像的像素级无人机检测。",
    "针对小目标易受背景噪声干扰的问题，围绕 group_length、channel_dim、patch_size 及目标先验光谱生成策略开展参数优化，并结合概率热力图与 ROC/AUC 指标评估检测效果，增强模型对细粒度光谱特征的表征与识别能力。",
]


def set_paragraph_text(paragraph, text):
    text_nodes = paragraph.xpath(".//w:t", namespaces=NS)
    if not text_nodes:
        run = etree.SubElement(paragraph, W + "r")
        text_node = etree.SubElement(run, W + "t")
    else:
        text_node = text_nodes[0]
        for extra in text_nodes[1:]:
            extra.getparent().remove(extra)
    text_node.text = text


def set_num_id(paragraph, num_id):
    num_ids = paragraph.xpath("./w:pPr/w:numPr/w:numId", namespaces=NS)
    if len(num_ids) != 1:
        raise RuntimeError("Expected one numbering reference in bullet template")
    num_ids[0].set(W + "val", str(num_id))


def add_numbering_instance(numbering_root, source_num_id):
    num_elements = numbering_root.xpath("./w:num", namespaces=NS)
    existing_ids = [int(node.get(W + "numId")) for node in num_elements]
    source = next(
        node for node in num_elements if node.get(W + "numId") == str(source_num_id)
    )
    new_num_id = max(existing_ids) + 1
    new_num = deepcopy(source)
    new_num.set(W + "numId", str(new_num_id))
    level_override = etree.SubElement(new_num, W + "lvlOverride")
    level_override.set(W + "ilvl", "0")
    start_override = etree.SubElement(level_override, W + "startOverride")
    start_override.set(W + "val", "1")
    numbering_root.append(new_num)
    return new_num_id


def main():
    with ZipFile(SOURCE) as source_zip:
        entries = {name: source_zip.read(name) for name in source_zip.namelist()}

    document_root = etree.fromstring(entries["word/document.xml"])
    numbering_root = etree.fromstring(entries["word/numbering.xml"])
    new_num_id = add_numbering_instance(numbering_root, source_num_id=8)

    targets = []
    for textbox in document_root.xpath(".//w:txbxContent", namespaces=NS):
        combined_text = "".join(textbox.xpath(".//w:t/text()", namespaces=NS))
        if "PX4" in combined_text and "MAVProxy" in combined_text:
            targets.append(textbox)

    if len(targets) != 2:
        raise RuntimeError(f"Expected modern and fallback project text boxes, found {len(targets)}")

    for textbox in targets:
        paragraphs = textbox.xpath("./w:p", namespaces=NS)
        if len(paragraphs) < 5:
            raise RuntimeError("Project text box does not contain the expected template paragraphs")

        title_paragraph = deepcopy(paragraphs[0])
        set_paragraph_text(title_paragraph, TITLE)

        new_paragraphs = [title_paragraph]
        for bullet_text in BULLETS:
            bullet_paragraph = deepcopy(paragraphs[1])
            set_paragraph_text(bullet_paragraph, bullet_text)
            set_num_id(bullet_paragraph, new_num_id)
            new_paragraphs.append(bullet_paragraph)

        insert_at = paragraphs[-1] if not "".join(
            paragraphs[-1].xpath(".//w:t/text()", namespaces=NS)
        ).strip() else None
        for new_paragraph in new_paragraphs:
            if insert_at is None:
                textbox.append(new_paragraph)
            else:
                textbox.insert(textbox.index(insert_at), new_paragraph)

    entries["word/document.xml"] = etree.tostring(
        document_root, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    entries["word/numbering.xml"] = etree.tostring(
        numbering_root, xml_declaration=True, encoding="UTF-8", standalone=True
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as output_zip:
        for name, data in entries.items():
            output_zip.writestr(name, data)

    print(OUTPUT)


if __name__ == "__main__":
    main()
