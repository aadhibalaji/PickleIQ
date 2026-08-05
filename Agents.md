# PickleIQ End-User Frontend Instructions

## Project purpose

Build the customer-facing frontend for PickleIQ, a pickleball video-analysis application.

This frontend is separate from the existing internal/testing frontend used by the development team. Do not replace, merge with, redesign, or remove the internal testing interface unless a task explicitly requests it.

The end-user flow is:

1. Upload a pickleball video.
2. Select Singles or Doubles.
3. Select Positioning or Form analysis.
4. Start analysis.
5. View Pass 1 rally clipping progress.
6. View Pass 2 rally filtering progress.
7. View player-analysis progress while the final trimmed video plays.
8. View general coaching feedback and timestamp-specific feedback.
9. Return home and reset the flow.

## Before editing

Before making changes:

1. Inspect the repository structure.
2. Identify the framework, routing approach, styling system, state-management approach, and existing testing frontend.
3. Determine the safest location for the new end-user frontend.
4. Reuse existing project conventions when they are suitable.
5. Do not assume backend endpoints, response fields, upload contracts, or processing events that are not present in the repository.

If backend integration is unavailable, create a clearly isolated mock/demo adapter for the end-user flow. Do not disguise mock behavior as a completed production integration.

## Architecture requirements

The end-user interface must have its own clear route or entry point.

Prefer a state-driven flow with named stages:

* `upload`
* `pass1`
* `pass2`
* `analysis`
* `results`

Do not implement the entire experience as one large component.

Separate:

* screen-level components
* shared buttons
* segmented toggles
* progress displays
* video presentation
* rally markers
* timestamp feedback
* processing/API adapters
* flow state

Keep processing logic separate from visual components.

## Scope protection

Do not modify the internal testing frontend unless required to establish routing or shared infrastructure.

Do not remove existing routes, utilities, API clients, environment handling, or test tools.

Avoid broad refactors unrelated to the current screen.

## Design principles

The product should feel:

* focused
* premium
* calm
* minimal
* analytical
* slightly warm rather than cold or neon

Avoid:

* generic dashboard styling
* excessive cards
* gradients unless explicitly requested
* glassmorphism
* bright purple styling
* gaming aesthetics
* crowded navigation
* decorative background noise
* unnecessary icons
* excessive instructional copy

## Responsiveness

The current target is laptop and desktop screens.

Support a sensible minimum desktop width, but do not spend task scope building a full mobile layout unless explicitly requested.

The interface should remain usable around common laptop widths such as 1280px and 1440px.

## Accessibility

Use semantic controls.

All interactive elements must be keyboard accessible and show a visible focus state.

Do not communicate state using color alone.

Use accessible labels for icon-only controls, including the upload control.

Maintain readable contrast against the dark background.

## Implementation quality

Use the project’s existing language and type system. If the project uses TypeScript, preserve strict typing.

Do not introduce a new dependency when a small existing or native solution is sufficient.

Do not create duplicate components when an appropriate shared component already exists.

Do not use placeholder lorem ipsum.

Do not claim tests or commands passed unless they were actually run.

## Verification

For each task:

1. Run the relevant formatter or linter.
2. Run targeted tests when they exist.
3. Run a production build or typecheck when practical.
4. Inspect the changed screen at the target desktop viewport.
5. Report:

   * files changed
   * implementation decisions
   * commands run
   * any unresolved backend or asset dependencies
