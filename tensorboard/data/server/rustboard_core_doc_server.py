from http import server
import mimetypes
import os
import zipfile

from werkzeug import exceptions
from werkzeug import serving
from werkzeug import utils
from werkzeug import wrappers


def main():
    webfiles = os.path.join(os.path.dirname(__file__), "rustboard_core_doc.zip")
    data = {}

    pfx = (
        "bazel-out/k8-fastbuild/bin/tensorboard/data/server/rustboard_core_doc/"
    )
    with open(webfiles, "rb") as fp:
        with zipfile.ZipFile(fp) as zp:
            for path in zp.namelist():
                if not path.startswith(pfx):
                    continue
                data[path[len(pfx) :]] = zp.read(path)

    @wrappers.Request.application
    def app(request):
        p = request.path.lstrip("/")
        if not p:
            return utils.redirect("/rustboard_core/index.html")
        if p.endswith("/"):
            p += "index.html"
        blob = data.get(p)
        if not blob:
            raise exceptions.NotFound()
        return wrappers.Response(blob, content_type=mimetypes.guess_type(p)[0])

    serving.run_simple("0.0.0.0", 6005, app)


if __name__ == "__main__":
    main()
