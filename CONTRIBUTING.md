
Licenses
========

ParadisEO modules are using free software licenses,
any contribution should be licensed under the same license.

| Module | License | Version | Copyleft | Patent-left |
|--------|---------|---------|----------|-------------|
| EO     | LGPL    |  2      | Lib only | No          |
| EDO    | LGPL    |  2      | Lib only | No          |
| MO     | CeCILL  |  2.1    | Yes      | No          |
| MOEO   | CeCILL  |  2.1    | Yes      | No          |
| SMP    | CeCILL  |  2.1    | Yes      | No          |


Contribution Workflow
=====================

The maintainer(s) will try to answer under a couple of weeks, if not, do not hesitate to send an e-mail.

If you're not familiar with Git and merge requests, start by cloning one of the main repository:
- `git clone https://github.com/nojhan/paradiseo.git`
- `git clone https://scm.gforge.inria.fr/anonscm/git/paradiseo/paradiseo.git`


Git workflow
------------

ParadisEO follows a classical Git workflow using merge requests.
In order to fix a bug or add a feature yourself, you would follow this process.

```bash
cd paradiseo
git pull origin master # Always start with an up-to-date version.
git checkout -b <my_feature> # Always work on a dedicated branch.
# [ make some modifications… ]
git commit <whatever>
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -BUILD_TESTING=ON -DENABLE_CMAKE_TESTING=ON .. && make && ctest # Always test.
cd ..
git pull origin master # Always check that your modification still merges.
```

If everything went without error, you can either send the patch or submit a merge request.
To do so, you can either:
- submit a "pull request" on Github: [nojhan/paradiseo](https://github.com/nojhan/paradiseo),
- or send a patch on the [ParadisEO mailing list](https://lists.gforge.inria.fr/cgi-bin/mailman/listinfo/paradiseo-users).

See below for the details.


Github pull request
-------------------

Once logged in Github, go to the [maintainer repository](https://github.com/nojhan/paradiseo) and click the "fork" button.
You should have your own copy of the ParadisEO project under your own name.
Then add it as an additional "remote" to your ParadisEO Git tree: `git remote add me <your own URL>`.
Then, checkout the branch holding the modifications you want to propose, check that it merges with the main repository
and push it on your own repository:
```bash
git checkout <my_feature>
git pull origin master
git push me <my_feature>
```

Then go to the maintainer's repository page, click on the "Pull request" tab, and on the "New pull request" button.
You should then select the maintainer's master branch on the left dropdown list, and your own `my_feature` on the right one.
Explain why the maintainer should merge your modifications and click the "Submit" button.


E-mail your patch
-----------------

Generate a patch file from the difference between your branch and a fresh master:
```bash
git pull origin master
git diff master <my_feature> > my_feature.patch
```

Then send the `my_feature.patch` (along with your explanations about why the maintainer should merge your modifications)
to the [mailing list](https://lists.gforge.inria.fr/cgi-bin/mailman/listinfo/paradiseo-users).

