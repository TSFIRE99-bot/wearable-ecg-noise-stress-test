# Request for clinical review: AF classifiers under real recorded ECG noise

Thank you for agreeing to read this manuscript. It is a measurement study about single-lead ECG atrial fibrillation (AF)
classifiers of the kind used in smartwatches and handheld recorders. It is not a device trial, and it makes no claim about
any commercial product. I would value your view on whether the clinical framing is right before I submit it to
*Physiological Measurement*.

Reading time: about 30–45 minutes for the parts listed below.

## What the study does, in brief

Four AF classifiers were developed on 8,528 handheld single-lead ECGs (CinC 2017): a simple RR-irregularity rule, two
models built on beat-timing and P-wave features, and a convolutional neural network (CNN) that reads the raw waveform.
Real noise recorded from electrodes (electrode motion, muscle artifact, baseline wander) was added to test recordings at
increasing strength. The same test was repeated on lead I of PTB-XL (21,799 clinical 12-lead ECGs) and checked on
ambulatory Holter recordings with natural noise (MIT-BIH Atrial Fibrillation Database).

The main finding: models with almost identical accuracy on clean recordings fail in **opposite clinical directions**
under heavy electrode motion. Models that rely on detected beats raise **false AF alarms**, because the beat detector adds
spurious beats that make the rhythm look irregular. The CNN instead **misses AF**. Training with noise reduces both
errors. Withholding low-quality recordings ("inconclusive") helps less than expected, withholds AF recordings more often
than others, and a quality threshold set on one device does not carry over to another.

## Where to look

- Abstract
- Introduction, last paragraph (the hypotheses)
- Results on operating points and prevalence (sensitivity, specificity, PPV and NPV at fixed thresholds)
- Discussion and Limitations

## Questions I would most like answered

1. **Which error matters more?** The paper describes false alarms as leading to unnecessary follow-up and loss of user
   trust, and missed AF as delaying diagnosis. Is that a fair summary for consumer screening, and does the balance
   change for symptomatic patients or for monitoring after an AF diagnosis?
2. **Atrial flutter.** Records with atrial flutter but no AF were counted as non-AF, following the CinC 2017 labels.
   Is that acceptable for a screening classifier, given that flutter is usually managed like AF for stroke prevention?
   (Excluding all records with a flutter statement changed the results by less than 0.01.)
3. **"Inconclusive" results.** When a device withholds a result, what typically happens next in your practice? Is it
   clinically worse that low-quality recordings were more often AF recordings?
4. **Operating points.** The paper reports performance at the decision threshold each model would ship with, and at a
   threshold chosen in advance to give at least 90% sensitivity. Are these the right operating points to report?
5. **Prevalence scenarios.** Positive and negative predictive values are shown for screening prevalences of 0.5–10%,
   labelled as hypothetical. Are these ranges realistic for the populations that use these devices?
6. **Overstatement.** Please mark any sentence that claims more clinical relevance than a bench study can support.

## Credit

If your comments shape the clinical interpretation substantially, I would be glad to discuss co-authorship in line with
the ICMJE criteria (which include approving the final version and agreeing to be accountable for the work). Otherwise,
with your permission, I will thank you by name in the acknowledgements.

Tejas Menneni, Columbus State Community College, Columbus, OH, USA
