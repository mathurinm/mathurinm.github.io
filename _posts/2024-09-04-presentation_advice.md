---
layout: post
title:  "Advice for presentations or how to avoid losing your audience"
date:   2024-09-05 00:00:00 +0200
permalink:  /:title/
---

Summary advice given after attending one day of M1 internship presentations at ENS.


**TLDR**: Never underestimate how easily and quickly you can lose the audience, and how hard it is for them to catch up afterwards.
<!-- A presentation is very different from writing a paper or report. -->
Your slides should only be a visual support for what you say.
**Technicalities like equations with heavy notation, tables, complex figures are the enemy.**
Instead, use plenty of examples, figures, simple terms and explanations.



## Content: choose the appropriate level of detail
First, select what you want to include in your presentation, and how you want to structure it.

- Choose what you include in the presentation; you do not want to shoehorn all the technical content of your paper/report.
**The presentation  is not the paper, it is here to highlight the most significant points only**, to make people want to read the paper for more details.
If you want, have backup slides (after the conclusion) with more details about some point, that you may use during the questions.
- Contrary to a paper, **slides do not need to be 100% rigorous**, you can (and should!) be fuzzy, informal, have handwavy notation: all this, in order to gain clarity and concision.
- When working on the presentation, ask yourself: what do I want somebody completely unaware of my research to remember at the end of the talk? What's my big message? **What are the points that really matter, and what is more of a technical detail** (that should be omitted)?
    Instead of a "Questions?" slides, end with a recap slide that contains the 3/4 high-level takeaways.
- Rule of thumb: use at most 10 lines per slide, and spend 1 min/slide. Going over these limits often means that you put too much; you will rush over the slides and lose everyone on slide 2.
- Announcing your plan is usually not very interesting (introduction, state-of-the-art, presentation of method, results...).
    Instead, starting with a simplified motivating example allows to get the point quickly. Do not start by introducing all your notation.
- "A mathematical presentation should content one proof and one joke, and the two should never be the same thing".
    If you show proofs, just show an overview, or detail the key innovative step or whatever, but **avoid writing down many lines of complex computations with heavy notation**.
- Avoid including results tables, they are really hard to parse. Most people that include them end up saying "and this table contains too much to read, but it shows that..."
- Algorithms are also very hard to parse; the comments are usually more useful than the formulas themselves. Present them saying "those are the important steps, here is what they do" in layman terms; "the main difference with existing algorithms is...", "the most costly part is this..." but do not ever go step by step through an algorithm



## Pay attention to form
Beyond its raw content, the visual aspect of your slide is a powerful tool to convey information easily.

- In your first slide, identify yourself and your lab/structure/school clearly, but also acknowledge the people you've worked with (joint work with.../internship under the supervision of...).
- **Avoid at all cost writing long complete sentences**: the audience will read them instead of listening to you.
    Write only the important parts, to support your talk.
    The real content of the talk is your speech, not the slides.
    Somehow, if one can understand the slides just by reading them, it means you do not really provide additional value being there to introduce them.
- **Hierarchize content visually**: use bold and boxes to show the audience the critical content of the slide, play with font size to distinguish the key part from the rest (typically, citations can be made smaller, enumerations too).
- Make your slides as light as possible: do not number figures (do not even put them in a figure environment), do not number equations, remove the weird navigation bar that latex puts by default. **Every useless information is a distraction from the interesting content**.
- Colors are very helpful; for example if you display a variation of a previous formula, colorize what has changed.
- Use all the space in the slide (not by putting more content, but by putting enough space between each item).
    This is a pain in beamer but it's mandatory, dull slides will lose your audience faster than anything else.
    If there is a single figure in your slide it should take the whole space, etc.

## Spend time on figures
Those are very important, as a good figure conveys a lot of information quickly.
- It's often necessary to **redo the figure specifically for your presentation** (e.g. at least changing the ratio to better fit the slide).
    Editing the figure, previously saved as pdf, with inkscape or some other tool is often useful in that case.
- **Put xlabels, ylabels, and big enough**. As an exercise, take one presentation of yours, project it on the screen on a slide with a figure, go to the back of the room and check if you're able to parse it (you won't).
- Title and captions are often not needed.
- A 1-line sentence to interpret the figure, below it (or in the slide title) is summarizes the message.



## Presenting style
- **Rehearse**: your slides should be the support for your speech; if when rehearsing you realize that you're talking about something that needs visual support, it helps you improve your slides.
- **Start slow** in the introduction/motivating example so that the people completely unfamiliar with your line of work at least get something out of it.
    We often underestimate how a significant fraction of the audience can be unfamiliar with our line of work.
    They'll appreciate being walked through in the first 3/5 min.
- **Make breaks**, emphasize the important parts (such as "if there is one think to remember from the talk it is this result/slide/figure...")
- **Do some recaps when moving from one part to another**, summarizing in simple terms: "so we've seen that the big problem when trying to do X is Y and now we're going to see how to solve it".
- Give orders of magnitude, everything that helps grasp the problem (instead of vague sentences such as "this costs a lot", "there are many parameters", "the dimension is high", "the algorithm takes a lot of time to run", etc.)
- Look at the audience rather than your computer or the slides.
    Having a pointer (its costs 10€) helps.
    Avoid remaining perfectly still for the whole presentation.

## Small but useful stuff
- Number your slides for easier communication (a member of the audience may note its number to ask a question later), but avoid the "3/n" format: if n is big this is bad for the audience morale.
    Similarly I find the progress bar on top of the slide, present in default templates, to be utterly depressing (my 2 cents).
- Citations in slide are tricky: if the format is [ABC24] or [1] then they provide no info to the audience; I strongly recommend "Author et al (2024)", and to put them in smaller font when they are not critical.
- Beware of colors in graphs, many projectors are old and render them incorrectly.

