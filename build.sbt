ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.3.4"

lazy val root = (project in file("."))
  .settings(
    name := "agent-cats",
    libraryDependencies ++= Seq(
      "org.typelevel" %% "cats-core" % "2.12.0", // Cats Core library
      "org.typelevel" %% "cats-effect" % "3.5.2" // Optional: Cats Effect for functional effects

    )
  )
