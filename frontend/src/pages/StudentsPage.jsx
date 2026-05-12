import { useEffect, useState } from "react";
import Layout from "../components/Layout";
import { api } from "../lib/api";

export default function StudentsPage({ user, onLogout }) {
  const [students, setStudents] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [editingStudent, setEditingStudent] = useState(null);

  const [name, setName] = useState("");
  const [studentNumber, setStudentNumber] = useState("");
  const [batch, setBatch] = useState("");
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);

async function loadStudents() {
  try {
    setLoading(true);

    const studentsData = await api.get(`/students?userId=${user.id}`);
    const reportsData = await api.get(`/reports?userId=${user.id}`);

    const savedStudents = studentsData.students || [];
    const reports = reportsData.reports || [];

    const studentsWithStats = savedStudents.map((student) => {
      const studentReports = reports.filter(
        (report) => report.studentId === student.id
      );

      const essayCount = studentReports.length;

      const scores = studentReports
        .map((report) => Number(report.score || report.summary?.score || 0))
        .filter((score) => score > 0);

      const averageScore =
        scores.length > 0
          ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1)
          : null;

      return {
        ...student,
        essayCount,
        averageScore,
      };
    });

    setStudents(studentsWithStats);
  } catch (error) {
    console.error(error);
    alert("Could not load students.");
  } finally {
    setLoading(false);
  }
}

  useEffect(() => {
    if (user?.id) loadStudents();
  }, [user]);

  function openAddStudent() {
    setEditingStudent(null);
    setName("");
    setStudentNumber("");
    setBatch("");
    setEmail("");
    setShowForm(true);
  }

  function openEditStudent(student) {
    setEditingStudent(student);
    setName(student.name || "");
    setStudentNumber(student.studentNumber || "");
    setBatch(student.batch || "");
    setEmail(student.email || "");
    setShowForm(true);
  }

  function closeForm() {
    setEditingStudent(null);
    setName("");
    setStudentNumber("");
    setBatch("");
    setEmail("");
    setShowForm(false);
  }

  async function handleSaveStudent(e) {
    e.preventDefault();

    if (!name.trim()) {
      alert("Student name is required.");
      return;
    }

    if (!studentNumber.trim()) {
      alert("Student ID is required.");
      return;
    }

    try {
      if (editingStudent) {
        const data = await api.put(`/students/${editingStudent.id}`, {
          name: name.trim(),
          studentNumber: studentNumber.trim(),
          batch: batch.trim(),
          email: email.trim(),
        });

        setStudents(
          students.map((student) =>
            student.id === editingStudent.id
              ? { ...student, ...data.student }
              : student
          )
        );
      } else {
        const data = await api.post("/students", {
          name: name.trim(),
          studentNumber: studentNumber.trim(),
          batch: batch.trim(),
          email: email.trim(),
          userId: user.id,
        });

        setStudents([data.student, ...students]);
      }

      closeForm();
    } catch (error) {
      console.error(error);
      alert("Could not save student.");
    }
  }

  async function handleDeleteStudent(studentId) {
    const confirmDelete = window.confirm(
      "Are you sure you want to delete this student?"
    );

    if (!confirmDelete) return;

    try {
      await api.delete(`/students/${studentId}`);
      setStudents(students.filter((student) => student.id !== studentId));
    } catch (error) {
      console.error(error);
      alert("Could not delete student.");
    }
  }

  return (
    <Layout user={user} onLogout={onLogout} title="Students">
      <div className="page-container">
        <div className="page-header">
          <div>
            <h1>Students</h1>
            <p>
              Manage student records and view each learner’s essay evaluation
              history.
            </p>
          </div>

          <button className="primary-btn" onClick={openAddStudent}>
            + Add student
          </button>
        </div>

        {showForm && (
          <div className="modal-overlay">
            <div className="modal-box">
              <h2>{editingStudent ? "Edit Student" : "Add Student"}</h2>

              <form onSubmit={handleSaveStudent}>
                <label>Name</label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Enter student name"
                />

                <label>Student ID</label>
                <input
                  type="text"
                  value={studentNumber}
                  onChange={(e) => setStudentNumber(e.target.value)}
                  placeholder="Example: ST001"
                />

                <label>Batch / Intake</label>
                <input
                  type="text"
                  value={batch}
                  onChange={(e) => setBatch(e.target.value)}
                  placeholder="Example: Intake 24 / Level 2"
                />

                <label>Email optional</label>
                <input
                  type="text"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Enter email if available"
                />

                <div className="modal-actions">
                  <button type="button" onClick={closeForm}>
                    Cancel
                  </button>

                  <button type="submit" className="primary-btn">
                    {editingStudent ? "Save Changes" : "Add"}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {loading ? (
          <div className="empty-box">
            <p>Loading students...</p>
          </div>
        ) : (
          <div className="students-grid">
            {students.length === 0 ? (
              <div className="empty-box">
                <p>No students added yet.</p>
              </div>
            ) : (
              students.map((student) => (
                <div className="student-card" key={student.id}>
                  <h3>{student.name}</h3>

                  <p>
                    {student.studentNumber || "No student ID"}
                    {student.batch ? ` • ${student.batch}` : ""}
                  </p>

                  {student.email && <p>{student.email}</p>}

                  <div className="student-stats">
                    <span>{student.essayCount || 0} essays</span>
                    <span>
                      {student.averageScore
                        ? `Avg ${student.averageScore}/10`
                        : "No score yet"}
                    </span>
                  </div>

                  <div className="student-actions">
                    <button
                      type="button"
                      onClick={() => openEditStudent(student)}
                    >
                      Edit
                    </button>

                    <button
                      type="button"
                      onClick={() => handleDeleteStudent(student.id)}
                    >
                      Delete
                    </button>
                  </div>

                  <button
                    className="outline-btn"
                    onClick={() =>
                      (window.location.href = `/students/${student.id}`)
                    }
                  >
                    Open dashboard
                  </button>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </Layout>
  );
}