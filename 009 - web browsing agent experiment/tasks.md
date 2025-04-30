Goal of this project is to produce an agent which will interact with websites.
In index.ts we will be able to define set of instructions and then LLM and Pupetter will interact to perform them on given website.
Progress will be persisted by saving screenshots of each step in ./screenshots

Below are tasks you need to perform to complete the project. You need to go task by task and implement changes, after each completed task you must stop and ask user for feedback before continuing. After successfully completing task you must update its status and also produce report which will represent all the knowledge you gathered during performing previous task. Reports will be stored as ordered list in memory.md, each point of ordered list will represent task id.

| Id | Task    | Status | 
| --- | -------- | ------- | 
| 1   | Prepare web-agent.ts file which will contain logic related to pupetter interacting with websited. It should contain functions which LLM can use to interact with web.  | Completed    |
| 2   | Prepare prompt describing LLM how pupetter API from web-agent.ts works. Prompt should be part of index.ts | New     |
| 3   | Prepare initial prompt describing a task for LLM - navigate to google.com, type AI in search box, search for results, navigate to first result. | New    |