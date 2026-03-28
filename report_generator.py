def generate_report(label, freq, noise, edges, similarity, score):

    report = "\nAuthenticity Analysis Report\n\n"

    # classification
    if label.lower() == "fake":
        report += "Image Classification Result:\n"
        report += "The AI detection model identified patterns commonly found in AI-generated or manipulated images.\n\n"
    else:
        report += "Image Classification Result:\n"
        report += "The image appears consistent with natural photographic patterns.\n\n"

    # frequency
    report += "Frequency Analysis:\n"
    if freq > 6:
        report += "Unusual frequency artifacts were detected. These patterns are commonly produced by generative AI models.\n\n"
    else:
        report += "Frequency patterns appear consistent with natural images.\n\n"

    # noise
    report += "Noise Pattern Analysis:\n"
    if noise < 4:
        report += "The image has unusually smooth noise distribution. Real camera images usually contain natural sensor noise.\n\n"
    else:
        report += "Noise distribution appears natural.\n\n"

    # edges
    report += "Structural Analysis:\n"
    if edges > 1500:
        report += "The system detected structural inconsistencies in edges, which can occur when objects are inserted or removed.\n\n"
    else:
        report += "No major structural inconsistencies were detected.\n\n"

    # semantic
    report += "Semantic Consistency Check:\n"
    if similarity < 20:
        report += "The visual content shows slight inconsistency with natural photographic scenes.\n\n"
    else:
        report += "Scene semantics appear natural.\n\n"

    report += "Final Conclusion:\n"

    if score > 50:
        report += f"There is a high probability the image was manipulated or generated using AI. Estimated probability: {score}%.\n"
    else:
        report += f"The image appears likely authentic. Estimated manipulation probability: {score}%.\n"

    return report