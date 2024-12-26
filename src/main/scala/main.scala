import cats.data.{ValidatedNel, NonEmptyList, StateT}
import cats.implicits._

// ---------------------------------------
// 1) DATA MODEL
// ---------------------------------------
case class Agent(
                  name: String,
                  stateMachine: StateMachine,
                  expectedMessages: Set[MessageType]
                )

case class StateMachine(
                         initialState: State,
                         transitions: Map[State, List[Transition]]
                       )

case class Transition(trigger: Trigger, nextState: State)
case class State(name: String)

sealed trait Trigger
case class MessageTrigger(msgType: MessageType) extends Trigger
case object TimeoutTrigger extends Trigger
case class ValueSetTrigger(valueName: String) extends Trigger

case class MessageType(name: String)

case class Channel(name: String, sender: Agent, receiver: Agent)

case class Resource(
                     name: String,
                     global: Boolean,
                     probabilityDistribution: Option[ProbabilityDistribution]
                   )

case class ProbabilityDistribution(name: String)

case class Model(
                  agents: List[Agent],
                  channels: List[Channel],
                  resources: List[Resource]
                )

// ---------------------------------------
// 2) VALIDATEDNEL APPROACH
// ---------------------------------------
// This approach uses ValidatedNel for parallel error accumulation.

object ValidatedChecks {
  // An alias for a ValidatedNel of String errors
  type ValidationResult[A] = ValidatedNel[String, A]

  // Main entry point for validations
  def validateModel(model: Model): ValidationResult[Model] =
    (
      validateAgents(model.agents),                      // returns ValidationResult[List[Agent]]
      validateChannels(model.channels, model.agents),    // returns ValidationResult[List[Channel]]
      validateResources(model.resources)                 // returns ValidationResult[List[Resource]]
    ).mapN((_, _, _) => model)

  // Validate all Agents in the model
  def validateAgents(agents: List[Agent]): ValidationResult[List[Agent]] =
    agents.traverse(validateAgent)

  // Validate a single Agent
  def validateAgent(agent: Agent): ValidationResult[Agent] =
    (
      validateStateMachine(agent.stateMachine),
      validateExpectedMessages(agent)
    ).mapN((_, _) => agent)

  // Validate a StateMachine
  def validateStateMachine(sm: StateMachine): ValidationResult[StateMachine] =
    (
      validateNoMissingStates(sm),
      validateNoConflictingTransitions(sm),
      validateReachabilityOfStates(sm)
    ).mapN((_, _, _) => sm)

  // Example check: The initialState should be among all defined states
  private def validateNoMissingStates(sm: StateMachine): ValidationResult[Unit] = {
    val allStates = sm.transitions.keySet ++ sm.transitions.values.flatten.map(_.nextState)
    if (!allStates.contains(sm.initialState))
      "Initial state is not defined or not reachable".invalidNel
    else
      ().validNel
  }

  // Example check: No two transitions in the same state should use the same trigger
  private def validateNoConflictingTransitions(sm: StateMachine): ValidationResult[Unit] = {
    val conflicts = sm.transitions.toList.flatMap { case (state, transitions) =>
      val triggers = transitions.map(_.trigger)
      val duplicates = triggers.diff(triggers.distinct)
      duplicates.map(d => s"State ${state.name} has conflicting transitions for trigger $d")
    }
    if (conflicts.nonEmpty) conflicts.mkString("; ").invalidNel else ().validNel
  }

  // Stub: Always succeeds. In a real scenario, you'd do BFS/DFS to ensure all states are reachable
  private def validateReachabilityOfStates(sm: StateMachine): ValidationResult[Unit] =
    ().validNel

  // Check that all message triggers in the state machine are in agent's expectedMessages
  private def validateExpectedMessages(agent: Agent): ValidationResult[Unit] = {
    val triggers = agent.stateMachine.transitions.values.flatten.map(_.trigger)
    val messageTriggers = triggers.collect { case MessageTrigger(msgType) => msgType }
    val undefined = messageTriggers.filterNot(agent.expectedMessages.contains)

    if (undefined.nonEmpty)
      s"Agent ${agent.name} expects undefined messages: ${undefined.map(_.name).mkString(", ")}".invalidNel
    else
      ().validNel
  }

  // Validate channels
  def validateChannels(channels: List[Channel], agents: List[Agent]): ValidationResult[List[Channel]] = {
    val agentNames = agents.map(_.name).toSet
    val invalidChannels = channels.filterNot(c =>
      agentNames.contains(c.sender.name) && agentNames.contains(c.receiver.name)
    )
    if (invalidChannels.nonEmpty)
      s"Channels with invalid endpoints: ${invalidChannels.map(_.name).mkString(", ")}".invalidNel
    else
      channels.validNel
  }

  // Validate resources
  def validateResources(resources: List[Resource]): ValidationResult[List[Resource]] = {
    (
      validateResourceDefs(resources),
      validateProbabilityDistributions(resources)
    ).mapN((_, _) => resources)
  }

  // Example resource checks
  private def validateResourceDefs(resources: List[Resource]): ValidationResult[Unit] =
    ().validNel

  private def validateProbabilityDistributions(resources: List[Resource]): ValidationResult[Unit] = {
    val invalid = resources.filter(r => r.probabilityDistribution.exists(!isPdfValid(_)))
    if (invalid.nonEmpty)
      s"Resources with invalid PDF: ${invalid.map(_.name).mkString(", ")}".invalidNel
    else
      ().validNel
  }

  private def isPdfValid(pd: ProbabilityDistribution): Boolean = {
    // Stub check. Implement real PDF logic
    true
  }
}

// ---------------------------------------
// 3) STATET + EITHER APPROACH
// ---------------------------------------
// This approach uses StateT over an Either[NonEmptyList[String], A] to
// track a ValidationContext (e.g. visited states) and return errors.

object StatefulChecks {
  // Error type and effect type
  type ErrorList = NonEmptyList[String]
  type ValidationEffect[A] = Either[ErrorList, A]

  // Our 'state' is some context we carry around
  case class ValidationContext(visitedStates: Set[State] = Set.empty)

  // StateT to thread ValidationContext, with Either for errors
  type CheckState[A] = StateT[ValidationEffect, ValidationContext, A]

  // Helper functions
  def raiseError[A](msg: String): CheckState[A] =
    StateT.liftF(Left(NonEmptyList.one(msg)))

  def pureResult[A](a: A): CheckState[A] =
    StateT.pure[ValidationEffect, ValidationContext, A](a)

  def addVisitedState(s: State): CheckState[Unit] =
    StateT.modify(ctx => ctx.copy(visitedStates = ctx.visitedStates + s))

  // Validation logic
  def validateModel(model: Model): CheckState[Model] = for {
    _ <- validateAgents(model.agents)
    _ <- validateChannels(model.channels, model.agents)
    _ <- validateResources(model.resources)
  } yield model

  def validateAgents(agents: List[Agent]): CheckState[Unit] =
    agents.traverse_(validateAgent)

  def validateAgent(agent: Agent): CheckState[Unit] = for {
    _ <- validateInitialState(agent.stateMachine)
    _ <- validateNoConflictingTransitions(agent.stateMachine)
    _ <- validateExpectedMessages(agent)
  } yield ()

  // Check if the initialState is found among transitions
  def validateInitialState(sm: StateMachine): CheckState[Unit] = for {
    _ <- if (
      sm.transitions.keySet.contains(sm.initialState) ||
        sm.transitions.values.flatten.map(_.nextState).toSet.contains(sm.initialState)
    ) pureResult(())
    else raiseError(s"Initial state ${sm.initialState.name} is not reachable")
    _ <- addVisitedState(sm.initialState)
  } yield ()

  def validateNoConflictingTransitions(sm: StateMachine): CheckState[Unit] = {
    val conflicts = sm.transitions.toList.flatMap { case (state, transitions) =>
      val triggers = transitions.map(_.trigger)
      val duplicates = triggers.diff(triggers.distinct)
      duplicates.map(d => s"State ${state.name} has conflicting transitions for trigger $d")
    }
    if (conflicts.nonEmpty) raiseError(conflicts.mkString("; "))
    else pureResult(())
  }

  def validateExpectedMessages(agent: Agent): CheckState[Unit] = {
    val triggers = agent.stateMachine.transitions.values.flatten.map(_.trigger)
    val messageTriggers = triggers.collect { case MessageTrigger(msgType) => msgType }
    val undefined = messageTriggers.filterNot(agent.expectedMessages.contains)
    if (undefined.nonEmpty)
      raiseError(s"Agent ${agent.name} expects undefined messages: ${undefined.map(_.name).mkString(", ")}")
    else pureResult(())
  }

  def validateChannels(channels: List[Channel], agents: List[Agent]): CheckState[Unit] = {
    val agentNames = agents.map(_.name).toSet
    val invalid = channels.filterNot { c =>
      agentNames.contains(c.sender.name) && agentNames.contains(c.receiver.name)
    }
    if (invalid.nonEmpty) raiseError(s"Channels with invalid endpoints: ${invalid.map(_.name).mkString(", ")}")
    else pureResult(())
  }

  def validateResources(resources: List[Resource]): CheckState[Unit] =
    pureResult(()) // Placeholder for real resource checks
}

// ---------------------------------------
// 4) MAIN: DEMO BOTH APPROACHES
// ---------------------------------------
@main def demoValidationApproaches(): Unit = {
  // Create a minimal example model
  val stateInit    = State("init")
  val stateProcess = State("process")

  val smA = StateMachine(
    initialState = stateInit,
    transitions  = Map(
      stateInit    -> List(Transition(MessageTrigger(MessageType("M1")), stateProcess)),
      stateProcess -> List(Transition(TimeoutTrigger, stateInit))
    )
  )
  val smB = StateMachine(stateInit, Map.empty)

  val agentA = Agent("AgentA", smA, Set(MessageType("M1"), MessageType("M2")))
  val agentB = Agent("AgentB", smB, Set(MessageType("M2")))

  val channel  = Channel("channel1", agentA, agentB)
  val resource = Resource("r1", global = true, None)

  val model = Model(
    agents    = List(agentA, agentB),
    channels  = List(channel),
    resources = List(resource)
  )

  // ---------------------------
  // A) ValidatedNel Approach
  // ---------------------------
  println("\n=== ValidatedNel Approach ===")
  val validatedResult = ValidatedChecks.validateModel(model)
  validatedResult match {
    case v if v.isValid =>
      println("VALID")
      println("Validated model: " + v)
    case i =>
      println("INVALID")
      println("Errors: " + i)
  }

  // ---------------------------
  // B) StateT + Either Approach
  // ---------------------------
  println("\n=== StateT + Either Approach ===")
  val initialCtx = StatefulChecks.ValidationContext()
  val stateTResult: Either[NonEmptyList[String], (StatefulChecks.ValidationContext, Model)] =
    StatefulChecks.validateModel(model).run(initialCtx)

  stateTResult match {
    case Right((ctx, _)) =>
      println("VALID")
      println("ValidationContext visited states: " + ctx.visitedStates)
    case Left(errors) =>
      println("INVALID")
      println("Errors: " + errors.toList.mkString(", "))
  }
}
