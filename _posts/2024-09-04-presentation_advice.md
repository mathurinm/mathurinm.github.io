---
layout: post
title:  "Advice for presentations or how to avoid losing your audience"
date:   2024-09-04 00:00:00 +0200
permalink:  /:title/
---

Summary advice given wrote after attending a day of M1 internship presentations at ENS.


**TLDR**: Never underestimate how easily you can lose the audience, and how hard it is then for them to catch up.
**Technicalities like equations with heavy notation, tables, complex figures are the enemy.**

## Content
- Choose what you put focus on; you do not want to horseshoe all the technical content of your paper/report.
People can refer to your paper if they want more details.
Have backup slides (after the conclusion) with more details about some point, if you feel that you may need it during the questions.
- Make the presentation asking yourself: what do I want somebody completely unaware of my research to remember?
- Rule fo thumb: Use at most 10 lines per slide, and spend 1 min/slide

## Form
- In your first slide, identify yourself and your structure clearly, but also acknowledge the people you've worked with.
- Announcing your plan is usually not very interesting (introduction, state-of-the-art, presentation of method, results...).
    Instead, it is useful to periodic recap when transitioning between parts.
    Even if you've lost someone, you allow them to catch up at this point.
- Avoid at all cost writing complete sentence: the audience will read them instead of listening to you.
    Write only the important parts, to support your talk.
    The real content of the talk is your speech, not the slides.
    somehow, if one can understand the slides just by reading them, it means you do not really provide additional value being there to introduce them.
- Make your slides as light as possible: do not number figures (do not even put them in a figure environment), do not number equations, remove the weird navigation bar that latex puts by default.
- Colors are very helpful; for example if you display a variation of a previous formula, colorized what has changed.
- Figures: those are very important, as a good figure carries a lot of information easily. It's often necessary to **redo the figure specifically for your presentation** (e.g. at least changing the ratio to better fit the slide).
    Editing the figure, previously saved as pdf, with inkscape or some other tool is often useful in that case.
    **Put xlabels, ylabels, and big enough**. Take one presentation of yours, project it on the screen on a slide with a figure, go to the back of the room and check if you're able to parse it (you won't).
    On the other hand, title and captions are often not needed.
    However a 1-line sentence to interpret the figure is appreciated
- Use bold and boxes to show the audience the critical content of the slide, play with font size to distinguish the key part from the rest.

## Presenting style
- Make break, emphasize the important parts (such as "if there is one think to remember from the talk it is this result/slide/figure...)
- Give orders of magnitude, everything that helps grasp the problem (instead of sentences such as "this costs a lot", "there are many parameters", "the dimension is high", "the algorithm takes a lot of time to run", etc.)
- Look at the audience rather than your computer or the screen. Having a pointer (its costs 10€) helps.

## Small but useful stuff
- Number your slides for easier communication (one may note its number to ask a question later), but avoid the "3/n" format: if n is big this is bad for the audience morale.
    Similarly I find the progress bar on top of the slide, present in default templates, to be utterly depressing (my 2 cents).

Use all the space in the slide (not by putting more content, but by putting enough space between each item. this is a pain in beamer but it's mandatory, dull slides will lose your audience faster than anything else). If there is a single figure in your slide it should take the whole space, etc.


Avoid tables, they are really hard to parse



Use a conclusion: what should somebody that did not understand remember about your talk?
Why is it interesting?

Rehearse: your slides should be the support for your talk; rehearsing will help you realize if something is missing.

You can have backup slides with additional content.
EXAMPLES

ALgorithms are also very hard to parse; the comments are usually more useful than the formulas themsevles. Present them saying "those are the important steps, here is what they do" in layman terms; the main difference with existing algorithms is... the most costly part is this... but do not ever go step by step through an algorithm

Avoid long enumerations.
You have the right to be informal and handwavy, to make oversimplification to gain clarity.

References: contrary
Title of slide is precious

Beware of colors in graph, many videoproj are old, using markers
