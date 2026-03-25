# Git 分支、备份与 PR 规范流程

本文档基于当前仓库的实际分支结构编写，目标是把下面三件事区分清楚：

1. 自己长期开发用哪个分支
2. 如何只把一部分内容拿去发 PR
3. 在 VS Code 里如何方便地切换分支继续写代码

当前仓库已经存在的关键分支如下：

1. `main`
2. `work/colqwen2-local`
3. `docs/colqwen2-setup`

---

## 一、三个分支各自负责什么

### 1. `main`

`main` 是干净基线分支，建议主要承担这几个职责：

1. 跟踪上游仓库 `upstream/main`
2. 作为新分支的起点
3. 保持尽量干净，避免混入个人实验内容

这不代表 `main` 不是你的分支。它当然属于你自己的 fork，也当然能用。
但在协作场景里，推荐把它当作“母版”或“基线”，而不是日常堆各种实验内容的地方。

如果把很多个人实验、模型文件、临时脚本直接堆到 `main`，后面会出现这些问题：

1. 很难同步上游更新
2. 很难整理出干净 PR
3. 很难区分哪些是自用内容，哪些是准备贡献给上游的内容

### 2. `work/colqwen2-local`

这是你自己的长期开发分支，适合放完整工作内容。

可以把它理解成“自己的实验室分支”，适合放：

1. 本地测试脚本
2. 中间态代码
3. 临时实验文件
4. 不一定准备提交给上游的内容
5. 需要先备份到自己仓库的完整成果

日常继续开发时，优先在这个分支上工作。

### 3. `docs/colqwen2-setup`

这是一个干净 PR 分支。

它的职责不是“长期开发”，而是“只承载准备交给上游 review 的那部分内容”。

当前这个分支只放了 `documents` 相关内容，因此适合拿去发 PR。

---

## 二、推荐的日常协作模式

推荐始终遵守下面这套结构：

1. `main` 用来同步上游和作为干净基线
2. `work/...` 用来保存自己的完整开发内容
3. `docs/...`、`feat/...`、`fix/...` 用来准备 PR

你可以把流程理解为：

1. 先在 `work/...` 上自由开发
2. 觉得某一部分成熟了
3. 再从 `main` 切一个干净分支
4. 只把那一部分拿过去
5. 用那个干净分支发 PR

这样做的好处是：

1. 自己的完整工作不会丢
2. PR 内容更小、更清晰
3. 上游更容易 review
4. 自己后续也更容易维护

---

## 三、当前仓库最推荐的工作方式

结合现在这几个分支，建议这样使用：

### 场景 A：你只是继续自己写代码

切到 `work/colqwen2-local`：

```bash
git switch work/colqwen2-local
```

这条命令的意思是：

1. 切换到你的长期开发分支
2. 后续新增代码、调试、实验，默认都放在这里

如果你已经在这个分支上，命令执行后不会有额外变化。

### 场景 B：你要同步上游最新代码

先切回 `main`，再同步：

```bash
git switch main
git fetch upstream
git rebase upstream/main
git push origin main
```

逐条解释如下：

1. `git switch main`
   切换回干净基线分支 `main`

2. `git fetch upstream`
   从原仓库抓取最新提交，但还不修改你当前文件

3. `git rebase upstream/main`
   把你自己的 `main` 对齐到上游最新 `main`

4. `git push origin main`
   把同步后的 `main` 推送到你自己的 fork

执行完后，你的 fork 的 `main` 和上游主线就保持一致了。

### 场景 C：你要准备一个新的 PR

先从干净的 `main` 派生新分支：

```bash
git switch main
git fetch upstream
git rebase upstream/main
git switch -c docs/新的分支名
```

例如：

```bash
git switch main
git fetch upstream
git rebase upstream/main
git switch -c docs/update-setup-guide
```

然后再从 `work/colqwen2-local` 中挑你想提交的内容过来。

如果只拿一个目录：

```bash
git checkout work/colqwen2-local -- documents
```

如果只拿一个文件：

```bash
git checkout work/colqwen2-local -- tools/test_colqwen2.py
```

这条命令的意思是：

1. 不切换分支
2. 只是把另一个分支上的指定文件或目录复制到当前分支
3. 这样可以做到“自己改很多，但 PR 只交一部分”

---

## 四、最常用 Git 命令与详细解释

### 1. 查看当前分支和文件状态

```bash
git status --short --branch
```

作用：

1. 显示当前所在分支
2. 显示哪些文件改了
3. 显示哪些文件已暂存、哪些未暂存

这是最应该养成习惯的命令之一。每次切分支、提交前后都可以先看一眼。

### 2. 查看本地所有分支

```bash
git branch -vv
```

作用：

1. 查看本地有哪些分支
2. 查看当前在哪个分支
3. 查看每个分支跟踪哪个远程分支

### 3. 切换到已有分支

```bash
git switch work/colqwen2-local
```

作用：切换到已有分支。

### 4. 新建并切换到新分支

```bash
git switch -c feat/my-change
```

作用：

1. 新建一个分支
2. 立即切过去

这是最常用的新建分支方式。

### 5. 暂存改动

```bash
git add 文件名
git add 目录名
git add .
```

作用：把当前改动加入“准备提交”的列表。

建议：

1. 只提交明确想提交的内容时，优先 `git add 文件名`
2. 确认当前改动全部都要提交时，再使用 `git add .`

### 6. 提交改动

```bash
git commit -m "docs: add setup guide"
```

作用：把已暂存内容写入本地历史。

常见提交前缀示例：

1. `docs:` 文档修改
2. `feat:` 新功能
3. `fix:` 修 bug
4. `refactor:` 重构
5. `wip:` 进行中的本地备份提交

### 7. 推送分支到自己的 fork

```bash
git push -u origin work/colqwen2-local
```

作用：

1. 把本地分支推到自己的 GitHub 仓库
2. 用 `-u` 建立跟踪关系

第一次推分支时建议带 `-u`。
以后继续推送同一分支时，通常直接执行：

```bash
git push
```

### 8. 从上游同步主线

```bash
git fetch upstream
git rebase upstream/main
```

作用：

1. 拉取原仓库最新提交
2. 让你当前分支基于上游最新主线重新对齐

注意：这套命令通常优先在 `main` 上执行。

---

## 五、VS Code 里如何自由切换分支

在 VS Code 里切换分支，常用有两种方式：

1. 图形界面方式
2. 终端命令方式

建议两种都掌握。

### 方式 1：点击左下角分支名

VS Code 左下角状态栏通常会显示当前分支名。

例如现在可能显示：

1. `docs/colqwen2-setup`
2. `work/colqwen2-local`
3. `main`

操作步骤：

1. 点击左下角分支名
2. 在弹出的分支列表里选择目标分支
3. 点击要切换的分支即可

适合场景：

1. 快速切换已存在分支
2. 不想手输命令时

### 方式 2：命令面板切换

操作步骤：

1. 按 `Ctrl+Shift+P`
2. 输入 `Git: Checkout to...`
3. 选择要切换的分支

如果要新建分支，也可以选择：

1. `Git: Create Branch...`

### 方式 3：在 VS Code 终端切换

直接在终端输入：

```bash
git switch main
git switch work/colqwen2-local
git switch docs/colqwen2-setup
```

这是最准确、最可控的方式。

推荐记住这几个常用命令：

```bash
git switch main
git switch work/colqwen2-local
git switch docs/colqwen2-setup
git status --short --branch
```

### 切换分支前的注意事项

如果当前分支还有未提交改动，Git 可能不允许你切走，或者切过去后让工作区状态变得混乱。

遇到这种情况，常用三种处理方式：

1. 先提交

```bash
git add .
git commit -m "wip: save current work"
```

2. 先临时保存到 stash

```bash
git stash push -u -m "temp save before switching branch"
```

之后再切分支：

```bash
git switch main
```

恢复 stash：

```bash
git stash list
git stash pop
```

3. 如果改动很小，而且两个分支都允许带过去，再直接切

但对于初学阶段，不建议频繁这样做，因为容易搞乱。

---

## 六、适合当前仓库的标准工作流模板

下面给出一套可以重复使用的模板。

### 模板 A：继续自己开发

```bash
git switch work/colqwen2-local
git status --short --branch
```

开始写代码后：

```bash
git add .
git commit -m "wip: continue local colqwen2 work"
git push
```

适用场景：

1. 继续做你自己的完整版本
2. 暂时不准备发 PR

### 模板 B：准备一个只包含部分内容的 PR

```bash
git switch main
git fetch upstream
git rebase upstream/main
git push origin main
git switch -c docs/new-pr-branch
git checkout work/colqwen2-local -- documents
git status --short --branch
git add documents
git commit -m "docs: update setup guide"
git push -u origin docs/new-pr-branch
```

适用场景：

1. work 分支里内容很多
2. 只想把其中一小部分交给上游

### 模板 C：从当前 PR 分支继续改文档

```bash
git switch docs/colqwen2-setup
git status --short --branch
```

编辑完后：

```bash
git add documents
git commit -m "docs: refine colqwen2 setup guide"
git push
```

适用场景：

1. 上游 review 后让你补文档
2. 你自己想继续完善当前这条文档 PR

---

## 七、当前这几个分支下，你应该怎么选

如果你的目标是继续本地开发完整功能：

```bash
git switch work/colqwen2-local
```

如果你的目标是继续完善文档 PR：

```bash
git switch docs/colqwen2-setup
```

如果你的目标是同步上游、重新开一个干净分支：

```bash
git switch main
```

可以把它们简单记成：

1. `main` = 干净基线
2. `work/...` = 自己长期开发
3. `docs/...` / `feat/...` = 提交给上游的干净分支

---

## 八、常见错误与建议

### 错误 1：直接在 `main` 上乱改

后果：

1. 主线变脏
2. 不容易同步 upstream
3. 不容易拆 PR

建议：

1. 把 `main` 当干净底座
2. 平时写代码优先切到 `work/...`

### 错误 2：一个分支既做长期开发，又直接拿去发 PR

后果：

1. PR 容易包含无关内容
2. reviewer 难以理解

建议：

1. 自己长期开发用 `work/...`
2. 提交上游另开干净分支

### 错误 3：切分支前不看状态

建议每次先执行：

```bash
git status --short --branch
```

### 错误 4：不确认 PR 比较的是哪个分支

发 PR 时一定确认：

1. base 是 `upstream/main`
2. compare 是你准备好的干净分支

---

## 九、一句话总结

当前仓库最推荐的工作方式是：

1. `main` 负责保持干净并同步上游
2. `work/colqwen2-local` 负责你自己的完整开发和备份
3. `docs/colqwen2-setup` 负责只包含文档内容的 PR

如果只是继续写代码，就切到：

```bash
git switch work/colqwen2-local
```

如果只是继续改当前这条文档 PR，就切到：

```bash
git switch docs/colqwen2-setup
```

如果准备同步上游或重新开干净分支，就切到：

```bash
git switch main
```