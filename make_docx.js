const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign,
} = require("docx");
const fs = require("fs");

const BODY_FONT = "Times New Roman";
const CODE_FONT = "Courier New";
const BODY_SIZE = 24; // 12pt
const CODE_SIZE = 20; // 10pt

const border = { style: BorderStyle.SINGLE, size: 1, color: "AAAAAA" };
const borders = { top: border, bottom: border, left: border, right: border };

// Helper: mixed text + inline code paragraph
function bodyPara(segments, opts = {}) {
  const runs = segments.map(seg => {
    if (seg.code) {
      return new TextRun({ text: seg.text, font: CODE_FONT, size: CODE_SIZE });
    }
    if (seg.bold) {
      return new TextRun({ text: seg.text, font: BODY_FONT, size: BODY_SIZE, bold: true });
    }
    return new TextRun({ text: seg.text, font: BODY_FONT, size: BODY_SIZE });
  });
  return new Paragraph({
    children: runs,
    spacing: { after: 160 },
    ...opts,
  });
}

// Parse inline backtick codes in a string → array of segments
function parse(str) {
  const parts = [];
  const re = /`([^`]+)`/g;
  let last = 0, m;
  while ((m = re.exec(str)) !== null) {
    if (m.index > last) parts.push({ text: str.slice(last, m.index) });
    parts.push({ text: m[1], code: true });
    last = m.index + m[0].length;
  }
  if (last < str.length) parts.push({ text: str.slice(last) });
  return parts;
}

// Parse bold (**...**) and inline code in a string
function parseRich(str) {
  const parts = [];
  const re = /\*\*([^*]+)\*\*|`([^`]+)`/g;
  let last = 0, m;
  while ((m = re.exec(str)) !== null) {
    if (m.index > last) parts.push({ text: str.slice(last, m.index) });
    if (m[1] !== undefined) parts.push({ text: m[1], bold: true });
    if (m[2] !== undefined) parts.push({ text: m[2], code: true });
    last = m.index + m[0].length;
  }
  if (last < str.length) parts.push({ text: str.slice(last) });
  return parts;
}

function cell(text, header = false) {
  return new TableCell({
    borders,
    width: { size: 1560, type: WidthType.DXA },
    shading: header ? { fill: "2E4057", type: ShadingType.CLEAR } : { fill: "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: [new TextRun({
        text,
        font: text.match(/^[0-9\.\s\|MBs]+$/) ? CODE_FONT : BODY_FONT,
        size: header ? 20 : 18,
        bold: header,
        color: header ? "FFFFFF" : "000000",
      })],
      alignment: header ? AlignmentType.CENTER : AlignmentType.CENTER,
    })],
  });
}

const tableData = [
  ["Precision", "Intra mean", "Inter mean", "Ratio", "Storage", "Time"],
  ["float64", "0.3545284390", "0.4207652705", "1.1868307988", "44.85 MB", "5.23 s"],
  ["float32", "0.3545284398", "0.4207652712", "1.1868307980", "22.42 MB", "4.84 s"],
  ["float16", "0.3545068247", "0.4207455755", "1.1868476040", "11.21 MB", "14.23 s"],
  ["int8",    "0.3550323015", "0.4212152955", "1.1864140069",  "5.61 MB", "4.98 s"],
];

const tableRows = tableData.map((row, i) =>
  new TableRow({ children: row.map(t => cell(t, i === 0)) })
);

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: BODY_FONT, size: BODY_SIZE } },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: BODY_FONT, size: 32, bold: true, color: "1A1A2E" },
        paragraph: { spacing: { before: 320, after: 200 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: BODY_FONT, size: 26, bold: true, color: "2C3E50" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 },
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children: [
      // Title
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun({
          text: "Numerical Precision and wav2vec Representations: A Distance Analysis",
          font: BODY_FONT, size: 32, bold: true,
        })],
      }),

      // Setup
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text: "Setup", font: BODY_FONT, size: 26, bold: true })],
      }),
      bodyPara(parse(
        "Representations were extracted using `facebook/wav2vec2-base` on the Russian\u2013French interference corpus (19 speakers, 25 lexical items, 7,299 word-level occurrences). Frame-level hidden states were aggregated via mean pooling, yielding one 768-dimensional vector per occurrence. Cosine distance was used throughout. The float64 computation serves as the reference baseline; representations were then converted to float32, float16, and 8-bit integer (symmetric per-tensor quantisation: `scale = max|X| / 127`, values clipped to `[-128, 127]`). For each precision level, the mean intra-speaker distance (same speaker, same word, different recordings) and mean inter-speaker distance (different speakers, same word) were computed, along with their ratio as a proxy for speaker separability."
      )),

      // Results
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text: "Results", font: BODY_FONT, size: 26, bold: true })],
      }),
      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [1504, 1504, 1504, 1504, 1505, 1505],
        rows: tableRows,
      }),
      new Paragraph({ children: [], spacing: { after: 160 } }),

      // Discussion
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text: "Discussion", font: BODY_FONT, size: 26, bold: true })],
      }),

      // Q1 bold heading
      bodyPara(parseRich("**Does lower precision merely perturb the values, or fundamentally alter the structure?**")),

      bodyPara(parse(
        "Figure 1 shows the KDE distributions of intra- and inter-speaker distances across all four precision levels. The four subplots are visually indistinguishable: the intra-speaker distribution peaks near 0.08 and the inter-speaker distribution near 0.15 in all cases, and the relative ordering between the two is strictly preserved throughout. This is the central finding \u2014 reduced precision does not alter the geometry of the representation space at a macroscopic level."
      )),
      bodyPara(parse(
        "Figure 2 reveals the subtleties. Ten decimal places are required to observe any difference between float32 and float64: their separability ratios agree to nine significant figures (1.1868307988 vs. 1.1868307980), confirming they are statistically indistinguishable. Float16 shows a slight decrease in both intra and inter mean distances. This is a systematic effect of its coarser 10-bit mantissa: truncation across 768-dimensional dot products biases cosine similarities upward, uniformly compressing the distance scale. Notably, this compression affects intra- and inter-speaker pairs in slightly different proportions, causing the separability ratio to marginally increase for float16. This does not imply better discrimination \u2014 it reflects a distortion of scale, not a genuine gain in separability."
      )),
      bodyPara(parse(
        "Figure 3 makes the error structure explicit. Float32 residuals are negligible (median ~5\u00d710\u207b\u00b9\u2070), consistent with random cancellation of rounding errors averaged over thousands of pairs. Float16 residuals are slightly wider (median ~\u22121.4\u00d710\u207b\u2075) and systematically negative, confirming the directional bias above. Int8 shows the largest spread (median ~+5.8\u00d710\u207b\u2074) with a positive bias: coarse quantisation systematically overestimates distances on average."
      )),

      // Q2 bold heading
      bodyPara(parseRich("**Are low-precision representations acceptable for large-scale speech processing?**")),

      bodyPara(parse(
        "For deployment (inference, approximate nearest-neighbour retrieval), float16 offers an excellent trade-off: 4\u00d7 storage reduction relative to float32, with negligible geometric distortion. The 2.67\u00d7 slowdown observed in Figure 4 for float16 is a hardware artefact of CPU execution \u2014 on modern GPUs with native float16 tensor cores (e.g., NVIDIA A100), this penalty disappears and float16 typically matches or exceeds float32 throughput. Int8 provides an 8\u00d7 storage reduction over float64 and is acceptable for approximate retrieval tasks, though its systematic overestimation of distances warrants caution in precision-sensitive analyses."
      )),
      bodyPara(parse(
        "For scientific analysis of speaker variability, float32 is safe without reservation. Float16 preserves group-level statistics reliably, but its ~10\u207b\u2075 per-distance bias could affect fine-grained significance tests or ranking tasks. Int8 should be used carefully: its quantisation error (~5\u00d710\u207b\u2074) represents roughly 0.7% of the intra/inter distance gap (~0.066), a non-trivial distortion for analyses of subtle phonetic differences."
      )),

      // Conclusion
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text: "Conclusion", font: BODY_FONT, size: 26, bold: true })],
      }),
      bodyPara(parse(
        "Reducing numerical precision perturbs distance values at all levels, but does not fundamentally alter the structure of the wav2vec2 representation space down to float16. The relative ordering of intra- and inter-speaker distances is fully preserved, and the separability ratio remains stable to four significant figures. Int8 quantisation introduces a larger but still moderate distortion, primarily a systematic upward bias. For large-scale speech processing, float32 is the safest choice at 2\u00d7 compression; float16 is acceptable on GPU hardware for most use cases; int8 warrants validation against a higher-precision baseline before use in precision-sensitive analyses."
      )),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/Users/zhangyawen/Desktop/proskills/lab5/data/final_report.docx", buf);
  console.log("Done.");
});
