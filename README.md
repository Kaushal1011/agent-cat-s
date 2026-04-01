# agent-cat-s 🐱

A Scala 3 library demonstrating **two functional validation patterns** for multi-agent systems using **Cats** and **Cats Effect**.

## Overview

This library validates agent-based models with state machines, channels, and resources. It showcases two complementary approaches to error handling:

1. **ValidatedNel** - Parallel error accumulation (all errors at once)
2. **StateT + Either** - Stateful validation with sequential error handling

## Features

✨ **Functional validation patterns** using Cats data types
🔍 **State machine validation** - detect conflicting transitions, unreachable states
📡 **Channel validation** - ensure agent endpoints exist
🎲 **Resource validation** - validate probability distributions
🧪 **Runnable examples** - see both approaches in action

## Quick Start

### Prerequisites

- Scala 3.3.4
- sbt 1.x

### Run the Demo

```bash
sbt run
```

This executes `demoValidationApproaches()` which validates a sample multi-agent model using both patterns.

## Validation Approaches

### 1️⃣ ValidatedNel Approach

**Best for:** Collecting all validation errors in parallel

```scala
val result: ValidatedNel[String, Model] = ValidatedChecks.validateModel(model)
result match {
  case Valid(m) => println(s"✓ Model valid: $m")
  case Invalid(errors) => println(s"✗ Errors: ${errors.toList.mkString(", ")}")
}
```

**Characteristics:**
- Accumulates **all errors** before returning
- Uses applicative composition (`mapN`, `traverse`)
- Great for form validation, config checks

### 2️⃣ StateT + Either Approach

**Best for:** Stateful validation with context tracking

```scala
val result = StatefulChecks.validateModel(model).run(ValidationContext())
result match {
  case Right((ctx, m)) => println(s"✓ Valid. Visited: ${ctx.visitedStates}")
  case Left(errors) => println(s"✗ Errors: ${errors.toList.mkString(", ")}")
}
```

**Characteristics:**
- Tracks **mutable context** (e.g., visited states) during validation
- Sequential error handling (fail-fast or continue)
- Useful for graph traversal, dependency checks

## Core Data Model

```scala
case class Agent(name: String, stateMachine: StateMachine, expectedMessages: Set[MessageType])
case class StateMachine(initialState: State, transitions: Map[State, List[Transition]])
case class Channel(name: String, sender: Agent, receiver: Agent)
case class Resource(name: String, global: Boolean, probabilityDistribution: Option[ProbabilityDistribution])
```

## Validation Rules

| Rule | Description |
|------|-------------|
| **No missing states** | Initial state must exist in transitions |
| **No conflicting transitions** | Same state can't have duplicate triggers |
| **Expected messages** | Message triggers must be in agent's expected set |
| **Valid channels** | Sender/receiver agents must exist |
| **Valid PDFs** | Resource probability distributions must be well-formed |

## Project Structure

```
agent-cat-s/
├── build.sbt              # Dependencies: cats-core, cats-effect
├── src/main/scala/
│   ├── main.scala         # Validation logic + demo
│   └── DataModel.scala    # (empty, reserved for future models)
└── README.md
```

## Dependencies

```scala
libraryDependencies ++= Seq(
  "org.typelevel" %% "cats-core" % "2.12.0",
  "org.typelevel" %% "cats-effect" % "3.5.2"
)
```

## Example Usage

```scala
// Define states and transitions
val stateInit = State("init")
val stateProcess = State("process")

val sm = StateMachine(
  initialState = stateInit,
  transitions = Map(
    stateInit -> List(Transition(MessageTrigger(MessageType("M1")), stateProcess)),
    stateProcess -> List(Transition(TimeoutTrigger, stateInit))
  )
)

val agent = Agent("MyAgent", sm, Set(MessageType("M1")))
val model = Model(agents = List(agent), channels = List(), resources = List())

// Validate with ValidatedNel
ValidatedChecks.validateModel(model) match {
  case Valid(_) => println("✓ All checks passed")
  case Invalid(errs) => println(s"✗ Found ${errs.size} errors")
}
```

## Contributing

Pull requests welcome! This is a learning/demo project showcasing functional patterns.

## License

MIT (assumed - add LICENSE file if needed)

## Learn More

- [Cats Documentation](https://typelevel.org/cats/)
- [Cats Effect](https://typelevel.org/cats-effect/)
- [Validated vs Either](https://typelevel.org/cats/datatypes/validated.html)