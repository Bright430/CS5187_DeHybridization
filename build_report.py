"""Assemble the report (English, IEEE Transactions format)."""

import os
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report_lib import (TEMPLATE, OUT, body, caption_note, clean, figure, fmt,
                        heading, math_runs, run, set_cols, table)

TITLE = "De-Hybridization: Can a Hybrid Image Be Taken Apart?"
AUTHOR = "Author Name"
AFFIL = ("Department of Computer Science, City University of Hong Kong, "
         "Hong Kong SAR, China")

ABSTRACT = (
    "A hybrid image fuses two faces into a single picture whose perceived identity "
    "depends on viewing distance: one person dominates at close range, the other from "
    "afar. The effect is usually demonstrated, rarely inverted. We ask whether such an "
    "image can be computationally taken apart. Using 30 identity pairs from CelebA-HQ, "
    "we synthesize hybrids with a fixed band split and calibrate the high-frequency "
    "gain so that the composite is identity-ambiguous when sharp. With a deep face "
    "recognizer used strictly as a distance ruler, we measure four things. First, the "
    "composite flips from the high-frequency person to the low-frequency person as "
    "viewing blur grows, crossing over at a blur of about 2.2 px (20 of 30 pairs). "
    "Second, each band alone identifies its own source almost perfectly (cosine 0.58 "
    "versus 0.11 for the low band, 0.49 versus 0.14 for the high band; 100% and 96.7% "
    "two-way accuracy), while the composite is the least certain of the three (66.7%). "
    "Third, phase randomization and face inversion collapse the band identity to "
    "chance, so the cue is structural rather than a low-level texture bias. Fourth, "
    "inverting the band operator inside a 150-dimensional eigenface subspace recovers "
    "only 0.10 and 0.25 cosine, below the raw band and close to a ceiling of 0.13 and "
    "0.33 that even ground-truth faces cannot exceed. Hybrid images are readable but "
    "not writable: the identity is present and separable in each band, yet a global "
    "linear face prior cannot rebuild it."
)

INDEX = ("Index Terms\u2014Hybrid images, frequency bands, face identity, "
         "inverse problems, subspace priors.")

REFS = [
    'A. Oliva, A. Torralba, and P. G. Schyns, "Hybrid images," ACM Trans. Graph., '
    'vol. 25, no. 3, pp. 527\u2013532, Jul. 2006.',

    'P. G. Schyns and A. Oliva, "Dr. Angry and Mr. Smile: When categorization flexibly '
    'modifies the perception of faces in rapid visual presentations," Cognition, '
    'vol. 69, no. 3, pp. 243\u2013265, 1999.',

    'P. J. Burt and E. H. Adelson, "The Laplacian pyramid as a compact image code," '
    'IEEE Trans. Commun., vol. 31, no. 4, pp. 532\u2013540, Apr. 1983.',

    'M. Turk and A. Pentland, "Eigenfaces for recognition," J. Cogn. Neurosci., '
    'vol. 3, no. 1, pp. 71\u201386, 1991.',

    'J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive angular margin loss '
    'for deep face recognition," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. '
    '(CVPR), 2019, pp. 4690\u20134699.',

    'A. Papadopoulos, T. Nguyen, E. Durmus, and N. Memon, "IllusionPIN: Shoulder-surfing '
    'resistant authentication using hybrid images," IEEE Trans. Inf. Forensics Security, '
    'vol. 12, no. 12, pp. 2875\u20132889, Dec. 2017.',

    'Y. Gandelsman, A. Shocher, and M. Irani, "Double-DIP: Unsupervised image '
    'decomposition via coupled deep-image-priors," in Proc. CVPR, 2019, pp. 11026\u201311035.',

    'X. Wang, Y. Li, H. Zhang, and Y. Shan, "Towards real-world blind face restoration '
    'with generative facial prior," in Proc. CVPR, 2021, pp. 9168\u20139177.',

    'S. Zhou, K. C. Chan, C. Li, and C. C. Loy, "Towards robust blind face restoration '
    'with codebook lookup transformer," in Proc. NeurIPS, 2022.',

    'P. P\u00e9rez, M. Gangnet, and A. Blake, "Poisson image editing," ACM Trans. Graph., '
    'vol. 22, no. 3, pp. 313\u2013318, Jul. 2003.',

    'A. Borji, "CNNs and Transformers perceive hybrid images similar to humans," '
    'arXiv:2203.11678, 2022.',
]


def title_block(doc):
    p = doc.add_paragraph(style="Title")
    run(p, TITLE, size=22)
    fmt(p, size=22, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=6)

    p = doc.add_paragraph()
    run(p, AUTHOR, size=11)
    fmt(p, size=11, align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=2)

    p = doc.add_paragraph()
    run(p, AFFIL, size=10, italic=True)
    fmt(p, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=10)


def full_width_figure(doc, path, width, caption):
    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    set_cols(sec, 1)
    figure(doc, path, width, caption)
    sec2 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_cols(sec2, 2)


def main():
    doc = Document(TEMPLATE)
    clean(doc)

    s1 = doc.sections[0]
    set_cols(s1, 1)
    add_page_number(s1)
    title_block(doc)

    # ---------------- abstract ----------------
    p = doc.add_paragraph(style="Abstract")
    run(p, "Abstract\u2014", size=9, bold=True, italic=True)
    math_runs(p, ABSTRACT, size=9)
    fmt(p, size=9, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=6, spacing=1.0)

    p = doc.add_paragraph()
    run(p, "Index Terms\u2014", size=9, bold=True, italic=True)
    run(p, INDEX.split("\u2014", 1)[1], size=9, bold=True)
    fmt(p, size=9, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=4, spacing=1.0)

    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    set_cols(sec, 2)

    # ---------------- I. introduction ----------------
    heading(doc, "I. Introduction")

    body(doc, "A hybrid image contains two pictures in one [1]. The low-pass band of "
              "one image is added to the high-pass band of another, so that a viewer "
              "standing close to the print sees the sharp, high-frequency face while a "
              "viewer standing across the room sees only the smooth, low-frequency face. "
              "The demonstration is one of the most reproduced figures in the perception "
              "literature, and it carries a practical payload: IllusionPIN [6] uses "
              "hybrid images to make a PIN pad legible only to the person holding the "
              "phone, on the assumption that an observer at the wrong distance sees "
              "nothing useful.")

    body(doc, "That assumption has never been tested from the attacker's side. If a "
              "hybrid image is intercepted, is the hidden face actually hidden? The "
              "question is not rhetorical. The low band of a hybrid is, by construction, "
              "a complete albeit blurred face; the high band is an edge map that still "
              "carries the geometry of the other face. A recognition system does not "
              "have to stand at a particular distance: it can filter the image itself.")

    body(doc, "We therefore invert the usual experiment. Instead of asking what a human "
              "sees at a given distance, we ask what remains in the signal, and whether "
              "the signal can be put back together. This splits into two questions that "
              "the literature tends to conflate. Q1 (readability): does each band still "
              "identify its own source? Q2 (writability): can a face prior invert the "
              "band operator and reconstruct the source image?")

    body(doc, "We answer both on a controlled stimulus set of 30 identity pairs, using a "
              "modern face recognizer as a ruler rather than as a model of human "
              "perception, and we add the controls that such a claim requires. The "
              "contributions are: (i) a paired, geometry-matched hybrid dataset in which "
              "the composite is calibrated to be identity-ambiguous when sharp, so that "
              "any later switch cannot be blamed on a biased gain; (ii) a band-resolved "
              "measurement showing that identity survives almost intact in both bands "
              "while the composite is the least informative of the three; (iii) phase and "
              "inversion controls that rule out a texture artefact; and (iv) a ceiling "
              "analysis showing that the failure of reconstruction is a property of the "
              "linear face prior, not of missing information.")

    # ---------------- II. related work ----------------
    heading(doc, "II. Related Work")

    body(doc, "Hybrid images were introduced by Oliva, Torralba and Schyns [1], building "
              "on the observation that rapid categorization relies on different spatial "
              "frequency bands depending on the task [2] and on the classic Laplacian "
              "pyramid [3]. The original paper characterizes the percept, not the "
              "invertibility of the construction; the bands are treated as channels of "
              "human vision rather than as a code that could be decoded.")

    body(doc, "On the recognition side, eigenfaces [4] established that a linear subspace "
              "of face images captures a large part of identity variation, and ArcFace [5] "
              "established the current standard for measuring identity: an embedding in "
              "which cosine similarity is a calibrated distance between persons. We use "
              "the latter only as a ruler. This distinction matters because a recent study "
              "shows that CNNs and Transformers reproduce human hybrid-image perception "
              "closely [11]; that result makes networks tempting proxies for observers, "
              "but it also means that a network's agreement with a human tells us nothing "
              "about whether the underlying signal is recoverable.")

    body(doc, "On the reconstruction side, the modern route is a generative prior. "
              "Double-DIP [7] decomposes a single image into two layers with coupled "
              "deep priors, and blind face restoration methods such as GFPGAN [8] and "
              "CodeFormer [9] invert severe degradations by projecting onto a learned "
              "face manifold. These methods are the natural tool for Q2, but they are "
              "also expensive and, more importantly, they hide the question: a strong "
              "prior can hallucinate a plausible face from almost any input, so a "
              "successful reconstruction does not prove that the band contained the "
              "identity. We therefore start from the weakest prior that could work, a "
              "linear subspace, and measure how far it gets. Poisson editing [10] is "
              "relevant only as the classical tool for seamless band recombination, "
              "which we do not need here because our bands are reconstructed "
              "independently.")

    # ---------------- III. method ----------------
    heading(doc, "III. Method")

    body(doc, "The pipeline has two branches (Fig. 1). The upper branch reproduces the "
              "classical demonstration and asks how the perceived identity of the "
              "composite changes with viewing blur. The lower branch takes the composite "
              "apart and asks what each band still contains and whether it can be "
              "inverted. Neither branch uses human observers; every judgement is made by "
              "the same recognizer, so differences between conditions are attributable to "
              "the stimulus and not to a change of judge.")

    full_width_figure(doc, "fig_pipeline.png", 7.0,
                      "Fig. 1.  Two-branch protocol. Top: the composite is blurred by a "
                      "simulated viewing distance and its identity is read out. Bottom: "
                      "the composite is split into bands, each band is inverted inside a "
                      "face subspace, and the result is scored against the true sources. "
                      "Only the recognizer judges, so all comparisons share one ruler.")

    heading(doc, "A. Stimuli", 2)

    body(doc, "We draw faces from a 1500-image CelebA-HQ subset. Each image is passed "
              "through a YuNet detector and aligned to 512 \u00d7 512 with the eye "
              "midpoint fixed at a canonical position, which removes translation, scale "
              "and roll; 1224 images survive detection. Two faces form a pair when their "
              "identity cosine is below 0.1 and their five-landmark geometry descriptor "
              "agrees to within 0.03 (all lengths normalized by inter-ocular distance). "
              "Thirty pairs are sampled; the mean identity cosine within a pair is "
              "\u22120.001 (maximum 0.072), so the two members are genuinely different "
              "people, and the mean geometry distance is 0.016, so the two faces occupy "
              "the same region of the frame. This matters: if the sources differed in "
              "pose the high band would leak a silhouette cue and the experiment would "
              "measure alignment rather than identity.")

    body(doc, "The composite is H = LP_{\u03c3}(M) + \u03b1 \u00b7 HP_{\u03c3}(E), where "
              "E is the high-frequency source, M the low-frequency source, "
              "LP_{\u03c3} a Gaussian low-pass with \u03c3 = 6 px at 512 resolution, and "
              "\u03b1 a gain that balances the two bands. The gain is not cosmetic. "
              "Unbalanced, the composite is simply one of the two faces and the "
              "demonstration collapses. We sweep \u03b1 and keep the value at which the "
              "composite is closest to identity-ambiguous when sharp: \u03b1 = 0.3 gives "
              "cosine 0.155 to E and 0.584 to M, \u03b1 = 0.5 gives 0.282 and 0.413, "
              "\u03b1 = 1.0 gives 0.468 and 0.247, and \u03b1 = 0.7 gives 0.377 and 0.320, "
              "a margin of \u22120.058 with 33% of pairs read as M at zero blur. All "
              "experiments below use \u03b1 = 0.7. The composite is stored at 8-bit "
              "grayscale, and the truth images are kept alongside it, so every "
              "measurement has a ground truth.")

    body(doc, "Viewing distance is simulated by an additional Gaussian blur "
              "\u03c3_{v} applied to the composite at test time, with "
              "\u03c3_{v} \u2208 {0, 2, 4, 6, 8, 12, 16} px. This is a distance proxy, "
              "not a physical distance: the mapping from \u03c3_{v} to metres depends on "
              "print size and acuity, and we report blur in pixels to keep the claim "
              "falsifiable. The band split used for analysis is the same operator that "
              "built the image, \u03c3 = 6, so the analysis is not favoured by a "
              "mismatched filter.")

    heading(doc, "B. Identity Ruler", 2)

    body(doc, "Identity is measured with ArcFace w600k_r50 [5] in ONNX form: a 112 "
              "\u00d7 112 crop aligned by five landmarks, a 512-dimensional embedding, "
              "and cosine similarity after L2 normalization. Two implementation details "
              "matter for validity. First, band images are largely edge maps and the "
              "detector frequently finds no face in them; rather than let detection "
              "failure decide the result, we reuse the alignment matrix estimated on the "
              "composite and apply it to every derived image, so all conditions are "
              "cropped identically. Second, each embedding is averaged over the image and "
              "its mirror, which removes the small left-right bias of the network. As a "
              "reference for what a cosine means here, two different persons score "
              "0.010 \u00b1 0.069 on this data, so values above roughly 0.15 are "
              "meaningful and values near zero are not.")

    full_width_figure(doc, "fig_qual.png", 7.0,
                      "Fig. 2.  One pair (index 23). Top row: the two sources, the "
                      "composite, and its low and high bands. Bottom row: the composite "
                      "at increasing viewing blur \u03c3_{v}. The bands are displayed with "
                      "their mean restored; the composite is shown as stored.")

    heading(doc, "C. Band Inversion and Its Ceiling", 2)

    body(doc, "For Q2 we restrict reconstructions to a face subspace. A basis of 150 "
              "eigenfaces is estimated from 280 aligned faces (Gram-matrix trick, so the "
              "cost is in the number of faces rather than pixels), and the estimate is "
              "x\u0302 = \u03bc + \u03a6c with c solving the ridge normal equations of "
              "A\u03bc + A\u03a6c \u2248 y, where A is the band operator (low-pass or "
              "identity minus low-pass) and y the observed band. The regularization is "
              "relative to the trace of the normal matrix, so the two bands are treated "
              "comparably even though their scales differ by orders of magnitude.")

    body(doc, "A reconstruction number alone is uninterpretable: it could be low because "
              "the band lacks information or because the basis is too weak. We therefore "
              "also report a ceiling, obtained by projecting the ground-truth source face "
              "onto the same subspace and scoring that projection. The ceiling bounds "
              "every linear-subspace reconstruction of that band, whatever the inversion "
              "algorithm, because no estimate can be closer to the truth than the "
              "projection of the truth itself. The gap between reconstruction and ceiling "
              "measures algorithmic loss; the gap between ceiling and the raw band "
              "measures prior loss.")

    heading(doc, "D. Controls", 2)

    body(doc, "A high-band edge map is an unusual input for a face recognizer, and a "
              "positive cosine could in principle come from generic face-likeness rather "
              "than from the individual. We therefore run two controls on the high band. "
              "In the phase-randomized control the Fourier amplitude spectrum is kept and "
              "the phase is replaced by uniform noise, which preserves the spectral "
              "statistics but destroys all structure. In the inversion control the source "
              "face is rotated by 180\u00b0 before the band is taken, which preserves the "
              "local statistics and the amount of edge energy but puts the face in an "
              "orientation the recognizer does not treat as a face. If the high band's "
              "identity came from texture or from edge density, these controls would keep "
              "it; if it comes from configuration, they will not.")

    # ---------------- IV. evaluation ----------------
    heading(doc, "IV. Evaluation")

    heading(doc, "A. Perceived Identity Versus Viewing Blur", 2)

    body(doc, "Table I and Fig. 3 give the central demonstration. When the composite is "
              "sharp, the two identities are almost equidistant (0.377 to E, 0.320 to M) "
              "and only a third of the pairs are read as the low-frequency person. As "
              "blur increases the high-frequency evidence disappears first: at "
              "\u03c3_{v} = 4 the margin has reversed (0.188 versus 0.494) and 97% of "
              "pairs are read as M; at \u03c3_{v} = 6 all 30 are. Interpolating per pair, "
              "20 of 30 pairs cross over, at a mean blur of 2.19 px (median 2.24). The "
              "remaining ten are already on the M side when sharp, which is what the "
              "\u03b1 calibration predicts (33% at zero blur).")

    table(doc, "I|Identity cosine as a function of simulated viewing blur (\u03c3 = 6, "
               "\u03b1 = 0.7, 30 pairs).",
          ["\u03c3_{v} (px)", "cos to E", "cos to M", "read as M (%)"],
          [["0", "0.377", "0.320", "33"],
           ["2", "0.319", "0.368", "50"],
           ["4", "0.188", "0.494", "97"],
           ["6", "0.106", "0.579", "100"],
           ["8", "0.068", "0.573", "100"],
           ["12", "0.040", "0.438", "100"],
           ["16", "0.024", "0.298", "100"]],
          [0.85, 0.85, 0.85, 0.95])

    body(doc, "Two features of the curve deserve comment. The transition is sharp: "
              "between \u03c3_{v} = 0 and \u03c3_{v} = 4 the margin moves by 0.36, which "
              "is most of its total range. And the curve is not monotone in the "
              "low-frequency identity either. Beyond \u03c3_{v} = 8 the cosine to M falls "
              "from 0.579 to 0.298 while the share read as M stays at 100%: at that point "
              "the recognizer is no longer choosing between two people, it is looking at "
              "a blur. A distance demonstration that stops at \u03c3_{v} = 8 would "
              "therefore overstate how robust the low-frequency identity is; the useful "
              "window is narrow and we report its end.")

    figure(doc, "fig_blur.png", 3.4,
           "Fig. 3.  Identity cosine versus viewing blur, mean \u00b1 s.e.m. over 30 "
           "pairs. The crossover (dotted) is at \u03c3_{v} \u2248 2.2 px. Shaded: the "
           "region in which the composite is still ambiguous.")

    heading(doc, "B. Where the Identity Lives", 2)

    body(doc, "Table II answers Q1. The low band alone identifies M at 0.579 against "
              "0.106 for the other person, and it does so for all 30 pairs. The high band "
              "alone identifies E at 0.485 against 0.140, correct for 29 of 30. Both "
              "bands are therefore readable, and this is not a marginal effect: the "
              "high-band margin of 0.345 is about five times the standard deviation of "
              "the chance distribution.")

    body(doc, "The composite, by contrast, is the least decisive of the three: 0.377 "
              "against 0.320, correct for only 20 of 30 pairs. Superposition does not "
              "strengthen either identity, it cancels them against each other. This is "
              "the quantitative version of the intuition behind hybrid images, and it is "
              "also the argument for decomposition: if the goal is to know who is in the "
              "picture, splitting the image is strictly better than looking at it.")

    table(doc, "II|Identity content of each component (\u03c3 = 6, \u03b1 = 0.7, 30 "
               "pairs). The last column is the two-way accuracy within a pair.",
          ["Component", "cos to own source", "cos to other", "Two-way acc. (%)"],
          [["LP \u2192 M", "0.579", "0.106", "100.0"],
           ["HP \u2192 E", "0.485", "0.140", "96.7"],
           ["H (composite)", "0.377", "0.320", "66.7"]],
          [0.9, 1.0, 0.8, 0.95])

    full_width_figure(doc, "fig_band.png", 7.0,
                      "Fig. 4.  Left: identity of each component against the person it "
                      "should contain and against the other person. Right: high-band "
                      "controls. Phase randomization and face inversion both return the "
                      "cosine to chance, so the band identity is carried by structure, "
                      "not by edge statistics.")

    heading(doc, "C. Controls", 2)

    body(doc, "The controls (Table III, Fig. 4 right) all but eliminate the concern that "
              "the recognizer is responding to edges as such. Keeping the amplitude "
              "spectrum and randomizing the phase drops the cosine from 0.485 to "
              "\u22120.008; rotating the face by 180\u00b0 before taking the band drops it "
              "to 0.005. Both are within one standard deviation of the chance level of "
              "0.010. Since neither control changes how much edge energy the image "
              "contains, the 0.485 must come from the spatial arrangement of that energy, "
              "that is, from face configuration.")

    table(doc, "III|High-band controls (30 pairs, mean \u00b1 s.d.).",
          ["Variant", "cos to true identity"],
          [["raw high band", "0.485 \u00b1 0.097"],
           ["phase randomized", "\u22120.008 \u00b1 0.045"],
           ["face inverted (180\u00b0)", "0.005 \u00b1 0.039"],
           ["chance (different persons)", "0.010 \u00b1 0.069"]],
          [1.7, 1.4], first_left=True)

    heading(doc, "D. Retrieval", 2)

    body(doc, "A closed-set retrieval test over the 60 source faces returns 100% rank-1 "
              "accuracy for the composite, the low band and the high band alike. We "
              "report this result mainly to disown it: the task is too easy, because "
              "different pairs are dissimilar enough that even a heavily degraded query "
              "retrieves its own pair. The informative measurement is the two-way choice "
              "within a pair, which is the last column of Table II and the metric we use "
              "throughout. Retrieval establishes that information is present; the "
              "two-way test establishes whose information it is.")

    heading(doc, "E. Reconstruction and Its Ceiling", 2)

    body(doc, "Table IV and Fig. 5 answer Q2, and the answer is negative. Inverting the "
              "low band through the subspace yields 0.104 against 0.011, far below the "
              "0.579 that the raw band already achieves; inverting the high band yields "
              "0.252 against 0.030, again below the raw 0.485. Reconstruction does not "
              "merely fail to improve the signal, it destroys a large part of it. "
              "Two-way accuracy falls from 100% to 73% for the low band and from 96.7% to "
              "73% for the high band.")

    table(doc, "IV|Reconstruction versus the subspace ceiling (30 pairs, 150 "
               "eigenfaces).",
          ["Band", "Method", "cos target", "cos other", "Acc. (%)"],
          [["low", "raw band", "0.579", "0.106", "100.0"],
           ["low", "subspace inverse", "0.104", "0.011", "73.3"],
           ["low", "ceiling (truth projected)", "0.129", "0.009", "80.0"],
           ["high", "raw band", "0.485", "0.140", "96.7"],
           ["high", "subspace inverse", "0.252", "0.030", "73.3"],
           ["high", "ceiling (truth projected)", "0.332", "0.006", "86.7"]],
          [0.5, 1.6, 0.75, 0.7, 0.7])

    body(doc, "The ceiling rows explain why, and they are the part of the experiment we "
              "did not expect. Projecting the true source face into the subspace and "
              "scoring that projection gives only 0.129 for M and 0.332 for E. The "
              "reconstruction is therefore not far from the best it could be: it reaches "
              "81% and 76% of the ceiling respectively. The bottleneck is the prior, "
              "not the algorithm and not the band. A 150-dimensional global linear basis "
              "cannot represent the configuration that distinguishes one face from "
              "another, and no amount of tuning inside that basis will change the bound.")

    figure(doc, "fig_recover.png", 3.3,
           "Fig. 5.  Reconstruction (dark), raw band (light) and subspace ceiling "
           "(pale). In both bands the reconstruction sits close to the ceiling and well "
           "below the raw band, so the loss is attributable to the prior.")

    body(doc, "This also closes a loophole in the earlier reasoning. One could argue that "
              "a better inversion, a different regularizer, or a deeper subspace would "
              "recover the face. The ceiling rules this out for the whole family: the "
              "reconstruction error is dominated by the projection error of the ground "
              "truth, which every member of the family shares. What remains open is the "
              "non-linear family, which we discuss next.")

    # ---------------- V. discussion ----------------
    heading(doc, "V. Discussion")

    body(doc, "The two questions have different answers, and the difference is the "
              "result. A hybrid image is readable: each band carries the identity of its "
              "own source almost as well as the source itself, and a simple two-way test "
              "recovers it reliably. It is not writable: no linear face prior we tried "
              "reconstructs a face whose identity survives the round trip, and the "
              "ceiling shows that the failure is structural rather than a matter of "
              "tuning.")

    body(doc, "The explanation is a null-space argument. The low-pass operator discards "
              "exactly the high-frequency configuration that distinguishes individuals, "
              "and the high-pass operator discards the low-frequency shading and shape "
              "that carry the rest. Each band therefore has a large null space, and the "
              "null space is not random with respect to identity: it is aligned with the "
              "directions in which faces differ. A global linear basis estimated by PCA "
              "spends its capacity on the directions of greatest variance across all "
              "faces, which are exactly the directions shared by everybody. The "
              "identity-bearing residual is small in variance and local in space, and "
              "eigenfaces are the wrong tool for it. This is consistent with the "
              "well-known weakness of linear subspace methods on unseen identities, and "
              "it explains why the low band, whose null space covers the whole "
              "high-frequency range, is hit harder than the high band.")

    body(doc, "Our reading is that the practical security claim behind hybrid-image "
              "authentication [6] survives, but for a different reason than usually "
              "given. The hidden face is not protected by being invisible; it is "
              "protected by being expensive to rebuild. An attacker with the composite "
              "and a recognizer can tell who the hidden person is, but cannot produce a "
              "usable image of them without a generative model. Whether a generative "
              "prior closes that gap is an empirical question we could not answer on "
              "CPU-only hardware, and it is the natural next step; the ceiling analysis "
              "suggests that such a model would have to be non-linear to help at all.")

    body(doc, "Several limitations bound these claims. The recognizer operates on 112 "
              "\u00d7 112 crops, so the measurement is itself band-limited: the highest "
              "frequencies of the high band are partly discarded before scoring, which "
              "can only understate the high-band identity. The recognizer is a ruler, not "
              "an observer; the 33% at zero blur is a property of our calibration and of "
              "ArcFace, not a prediction about human viewers, and we make no claim about "
              "what a person sees. The stimuli are grayscale, so colour cues are absent, "
              "and 30 pairs is enough for the effect sizes reported here but not for "
              "fine-grained stratification by age, gender or ethnicity, which we did not "
              "annotate. Only one band split (\u03c3 = 6) is used for the band analysis, "
              "although the dataset contains five; the trend across \u03c3 is consistent "
              "but we report the single setting to keep the comparisons clean. Finally, "
              "the linear prior is deliberately weak. A diffusion model would very likely "
              "produce a more convincing face, but a convincing face is not evidence that "
              "the band contained that identity, and separating the two would require an "
              "identity-preservation test rather than a visual one.")

    # ---------------- VI. conclusion ----------------
    heading(doc, "VI. Summary and Conclusions")

    body(doc, "We asked whether a hybrid image can be taken apart, and found that the "
              "answer depends on what \u201capart\u201d means. In terms of identification, "
              "yes, and easily: each band identifies its own source at 0.58 and 0.49 "
              "cosine with 100% and 96.7% two-way accuracy, the composite is the least "
              "informative of the three, and the cue survives neither phase "
              "randomization nor face inversion, which places it in face configuration "
              "rather than in edge statistics. In terms of reconstruction, no: a "
              "150-dimensional eigenface prior recovers 0.10 and 0.25 cosine, below the "
              "raw bands and within reach of a ceiling of 0.13 and 0.33 that even "
              "ground-truth faces cannot exceed under the same basis.")

    body(doc, "The lesson we find most interesting is methodological. The demonstration "
              "that a hybrid image deceives a viewer at one distance invites the "
              "conclusion that the information is gone at that distance. It is not gone; "
              "it is in the other band, and a machine that can filter does not have to "
              "move. Conversely, the fact that the information is present does not mean "
              "it can be reassembled, because reassembly requires a prior that knows what "
              "individual faces look like, and the cheap priors do not. Readable but not "
              "writable is a sharper statement than either \u201chidden\u201d or "
              "\u201crecoverable\u201d, and it is the one the measurements support.")

    body(doc, "Natural extensions are a non-linear prior with an identity-preservation "
              "test rather than a visual one, a colour version of the same protocol, and "
              "a human study that would let the machine ruler be calibrated against "
              "observers instead of standing in for them. The code, stimuli and result "
              "tables are available from the author.")

    # ---------------- references ----------------
    heading(doc, "References")
    for i, r in enumerate(REFS, start=1):
        p = doc.add_paragraph(style="References")
        run(p, f"[{i}]\t", size=8)
        run(p, r, size=8)
        fmt(p, size=8, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=2, spacing=1.0)
        p.paragraph_format.left_indent = Pt(14)
        p.paragraph_format.first_line_indent = Pt(-14)
        # the template style carries automatic numbering; switch it off
        pPr = p._p.get_or_add_pPr()
        numPr = pPr.makeelement(qn("w:numPr"), {})
        ilvl = pPr.makeelement(qn("w:ilvl"), {qn("w:val"): "0"})
        numId = pPr.makeelement(qn("w:numId"), {qn("w:val"): "0"})
        numPr.append(ilvl)
        numPr.append(numId)
        pPr.append(numPr)

    doc.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
